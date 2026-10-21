from __future__ import annotations

import os
import time
from functools import wraps

import torch


def install_canonical_zero12_reduction(logger, engine_class=None) -> bool:
    if os.environ.get("AURAL_DS_CANONICAL_REDUCTION", "0") != "1":
        return False
    if engine_class is None:
        from deepspeed.runtime.engine import DeepSpeedEngine

        engine_class = DeepSpeedEngine
    if getattr(engine_class, "_aural_canonical_zero12_reduction_installed", False):
        return True
    if not hasattr(engine_class, "coalesce_grad_reduction"):
        raise RuntimeError(
            "AURAL_DS_CANONICAL_REDUCTION=1 requires DeepSpeedEngine.coalesce_grad_reduction"
        )
    original_backward = engine_class.backward

    @wraps(original_backward)
    def canonical_backward(self, loss, retain_graph=False, scale_wrt_gas=True):
        stage = int(self.zero_optimization_stage())
        active = getattr(self, "_aural_canonical_backward_active", False)
        if stage not in (1, 2) or active:
            return original_backward(
                self, loss, retain_graph=retain_graph, scale_wrt_gas=scale_wrt_gas
            )
        accumulation_steps = int(self.gradient_accumulation_steps())
        if accumulation_steps != 1:
            raise RuntimeError(
                f"AURAL_DS_CANONICAL_REDUCTION=1 currently requires gradient_accumulation_steps=1; got {accumulation_steps}."
            )
        self._aural_canonical_backward_active = True
        try:
            with self.coalesce_grad_reduction():
                return original_backward(
                    self, loss, retain_graph=retain_graph, scale_wrt_gas=scale_wrt_gas
                )
        finally:
            self._aural_canonical_backward_active = False

    engine_class.backward = canonical_backward
    engine_class._aural_canonical_zero12_reduction_installed = True
    logger.info(
        "[qwen25-aural] installed canonical optimizer-order DeepSpeed ZeRO-1/2 gradient reduction"
    )
    return True


def install_fp32_grad_norm(logger) -> bool:
    if os.environ.get("AURAL_DS_FP32_GRAD_NORM", "1") != "1":
        return False
    from deepspeed.runtime.zero import stage_1_and_2 as zero

    optimizer_class = zero.DeepSpeedZeroOptimizer
    if getattr(optimizer_class, "_aural_fp32_grad_norm_installed", False):
        return True

    def get_grad_norm_direct(self, gradients, params, norm_type=2):
        norm_type = float(norm_type)
        selected = []
        for gradient, parameter in zip(gradients, params):
            if hasattr(parameter, zero.PIPE_REPLICATED) and parameter.ds_pipe_replicated:
                continue
            if zero.is_model_parallel_parameter(parameter) or self.model_parallel_rank == 0:
                selected.append(gradient.data.detach())
        if selected:
            flat_gradients = self.flatten(selected)
        else:
            flat_gradients = None
        chunk_elements = 16 * 1024 * 1024
        if norm_type == zero.inf:
            if selected:
                chunk_norms = [
                    chunk.abs().max().float() for chunk in flat_gradients.split(chunk_elements)
                ]
                total_norm = torch.stack(chunk_norms).max()
            else:
                total_norm = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            zero.dist.all_reduce(total_norm, op=zero.dist.ReduceOp.MAX, group=self.dp_process_group)
            self._model_parallel_all_reduce(tensor=total_norm, op=zero.dist.ReduceOp.MAX)
        else:
            if selected:
                total_norm = torch.zeros((), dtype=torch.float32, device=flat_gradients.device)
                for chunk in flat_gradients.split(chunk_elements):
                    values = chunk.float()
                    if norm_type == 2.0:
                        total_norm.add_(values.square().sum())
                    else:
                        total_norm.add_(values.abs().pow(norm_type).sum())
            else:
                total_norm = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            zero.dist.all_reduce(total_norm, op=zero.dist.ReduceOp.SUM, group=self.dp_process_group)
            self._model_parallel_all_reduce(tensor=total_norm, op=zero.dist.ReduceOp.SUM)
            total_norm = total_norm.pow(1.0 / norm_type)
        zero.mask_nan_or_inf_with_val_inplace(total_norm, device=self.device)
        return total_norm

    def scaled_global_norm(self, norm_type=2):
        if float(norm_type) != 2.0:
            raise ValueError("chunked ZeRO gradient norm only supports L2")
        profile = not getattr(self, "_aural_grad_norm_profiled", False)
        if profile:
            torch.cuda.synchronize(device=self.device)
            local_started = time.perf_counter()
        total_squared_norm = torch.zeros((), dtype=torch.float32, device=self.device)
        group_elements = []
        chunk_elements = 16 * 1024 * 1024
        for index, _group in enumerate(self.bit16_groups):
            selected = []
            for gradient, parameter in zip(
                self.averaged_gradients[index], self.params_in_partition[index]
            ):
                if hasattr(parameter, zero.PIPE_REPLICATED) and parameter.ds_pipe_replicated:
                    continue
                if zero.is_model_parallel_parameter(parameter) or self.model_parallel_rank == 0:
                    selected.append(gradient.data.detach())
            elements = sum((gradient.numel() for gradient in selected))
            group_elements.append(elements)
            if not selected:
                continue
            flat_gradients = self.flatten(selected)
            for chunk in flat_gradients.split(chunk_elements):
                values = chunk.float()
                total_squared_norm.add_(values.square().sum())
        if profile:
            torch.cuda.synchronize(device=self.device)
            local_elapsed = time.perf_counter() - local_started
            if zero.dist.get_rank() == 0:
                logger.info(
                    "[qwen25-aural] first chunked grad norm local: group_elements=%s total=%s seconds=%.4f",
                    group_elements,
                    sum(group_elements),
                    local_elapsed,
                )
            reduce_started = time.perf_counter()
        zero.dist.all_reduce(
            total_squared_norm, op=zero.dist.ReduceOp.SUM, group=self.dp_process_group
        )
        self._model_parallel_all_reduce(tensor=total_squared_norm, op=zero.dist.ReduceOp.SUM)
        total_norm = total_squared_norm.sqrt()
        zero.mask_nan_or_inf_with_val_inplace(total_norm, device=self.device)
        if profile:
            torch.cuda.synchronize(device=self.device)
            if zero.dist.get_rank() == 0:
                logger.info(
                    "[qwen25-aural] first chunked grad norm collective: seconds=%.4f",
                    time.perf_counter() - reduce_started,
                )
            self._aural_grad_norm_profiled = True
        return total_norm

    optimizer_class.get_grad_norm_direct = get_grad_norm_direct
    optimizer_class.scaled_global_norm = scaled_global_norm
    optimizer_class._aural_fp32_grad_norm_installed = True
    logger.info(
        "[qwen25-aural] installed chunked-flat FP32 DeepSpeed ZeRO gradient norm with one DP reduction"
    )
    return True
