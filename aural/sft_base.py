import os
from functools import wraps
from types import MethodType
from typing import Tuple

import torch
from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
from swift.utils import get_logger

from .audio import resolve_audio_token_id
from .latent_policy import ChunkLatentPolicy, compute_embeds_std
from .markers import EXPECTED_THINK_OPEN_IDS

logger = get_logger()


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return str(v).lower() in ("1", "true", "yes", "y")


class LatentSFTBase(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aural_max_r = _env_int("AURAL_COMPRESSION_FACTOR", 4)
        self.aural_fixed_r = _env_int("AURAL_COMPRESSION_FACTOR", 4)
        self.aural_ce_weight = _env_float("AURAL_CE_WEIGHT", 1.0)
        self.aural_embed_weight = _env_float("AURAL_EMBED_WEIGHT", 1.0)
        self.aural_entropy_weight = 0.0
        self.aural_embed_loss = "chunk_nll"
        self.aural_sqrt_mean = True
        self.aural_gmm_components = _env_int("AURAL_GMM_COMPONENTS", 4)
        self.aural_gmm_min_log_std = _env_float("AURAL_GMM_MIN_LOG_STD", -4.0)
        self.aural_gmm_max_log_std = _env_float("AURAL_GMM_MAX_LOG_STD", 2.0)
        self.aural_lp_lr = _env_float("AURAL_LP_LR", float(self.args.learning_rate))
        self.aural_latent_ce_soft = True
        self.aural_pef_weight = 1.0
        self.aural_pef_temperature = 1.0
        self.aural_pef_warmup_steps = _env_int("AURAL_LSS_WARMUP_STEPS", 500)
        self.aural_chunk = _env_int("AURAL_CHUNK_SIZE", 10)
        self.aural_lowrank = _env_int("AURAL_LOWRANK", 16)
        self.aural_mixer_mult = _env_int("AURAL_MIXER_MULT", 1)
        if self.aural_chunk < 1:
            raise ValueError(f"AURAL_CHUNK must be positive, got {self.aural_chunk}")
        if self.aural_lowrank < 0:
            raise ValueError(f"AURAL_LOWRANK must be non-negative, got {self.aural_lowrank}")
        if self.aural_mixer_mult < 1:
            raise ValueError(f"AURAL_MIXER_MULT must be positive, got {self.aural_mixer_mult}")
        self.think_open_id = EXPECTED_THINK_OPEN_IDS[0]
        self.think_close_id = EXPECTED_THINK_OPEN_IDS[0]
        self.aural_pad_id = self.tokenizer.pad_token_id
        newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        self.aural_newline_id = newline_ids[0] if len(newline_ids) == 1 else None
        base_model = self.template.get_base_model(self.model)
        thinker = getattr(base_model, "thinker", base_model)
        self._thinker = thinker
        self._text_model = thinker.model
        self._lm_head = thinker.lm_head
        self._embed_tokens = thinker.get_input_embeddings()
        self._audio_token_id = resolve_audio_token_id(thinker, self.tokenizer, base_model)
        hidden_size = self._embed_tokens.weight.shape[1]
        self.aural_hidden_size = hidden_size
        self.aural_lp_intermediate = _env_int("AURAL_LP_INTERMEDIATE", hidden_size)
        self.embeds_std = compute_embeds_std(thinker)
        latent_policy = ChunkLatentPolicy(
            feature_size=hidden_size,
            intermediate_size=self.aural_lp_intermediate,
            num_components=self.aural_gmm_components,
            chunk_size=self.aural_chunk,
            rank=self.aural_lowrank,
            mixer_mult=self.aural_mixer_mult,
            min_log_std=self.aural_gmm_min_log_std,
            max_log_std=self.aural_gmm_max_log_std,
        )
        dev = self._embed_tokens.weight.device
        latent_policy = latent_policy.to(device=dev, dtype=torch.float32)
        for p in latent_policy.parameters():
            p.requires_grad_(True)
        self.model.latent_policy = latent_policy
        self.latent_policy = latent_policy
        self._install_aural_forward()
        logger.info(
            f"[aural] max_r={self.aural_max_r} fixed_r={self.aural_fixed_r} ce_w={self.aural_ce_weight} embed_w={self.aural_embed_weight} embed_loss={self.aural_embed_loss}  chunk={self.aural_chunk} lowrank={self.aural_lowrank} mixer_mult={self.aural_mixer_mult} embeds_std={self.embeds_std:.5f} hidden={hidden_size} lp_inter={self.aural_lp_intermediate} gmm_components={self.aural_gmm_components} gmm_log_std=[{self.aural_gmm_min_log_std}, {self.aural_gmm_max_log_std}] latent_ce_soft={self.aural_latent_ce_soft} pef_weight={self.aural_pef_weight} pef_temp={self.aural_pef_temperature} pef_warmup={self.aural_pef_warmup_steps}"
        )
        logger.info(
            "[aural] latent policy parameters=%s",
            sum((p.numel() for p in latent_policy.parameters())),
        )

    def create_optimizer(self, *args, **kwargs):
        optimizer = super().create_optimizer(*args, **kwargs)
        if optimizer is None:
            return optimizer
        if getattr(self, "_aural_optimizer_grouped", False):
            return optimizer
        policy_ids = {
            id(parameter)
            for parameter in self.latent_policy.parameters()
            if parameter.requires_grad
        }
        grouped_ids = set()
        new_groups = []
        for group in optimizer.param_groups:
            policy_parameters = [
                parameter for parameter in group["params"] if id(parameter) in policy_ids
            ]
            if not policy_parameters:
                continue
            group["params"] = [
                parameter for parameter in group["params"] if id(parameter) not in policy_ids
            ]
            policy_group = {key: value for key, value in group.items() if key != "params"}
            policy_group["params"] = policy_parameters
            policy_group["lr"] = self.aural_lp_lr
            if "initial_lr" in policy_group:
                policy_group["initial_lr"] = self.aural_lp_lr
            new_groups.append(policy_group)
            grouped_ids.update((id(parameter) for parameter in policy_parameters))
        missing = [
            parameter
            for parameter in self.latent_policy.parameters()
            if parameter.requires_grad and id(parameter) not in grouped_ids
        ]
        if missing:
            new_groups.append(
                {"params": missing, "lr": self.aural_lp_lr, "weight_decay": self.args.weight_decay}
            )
        for group in new_groups:
            optimizer.add_param_group(group)
        self._aural_optimizer_grouped = True
        logger.info(
            "[aural] latent_policy optimizer groups=%s params=%s lr=%s backbone_lr=%s",
            len(new_groups),
            len(policy_ids),
            self.aural_lp_lr,
            self.args.learning_rate,
        )
        return optimizer

    def _install_aural_forward(self):
        if getattr(self.model, "_aural_forward_installed", False):
            return
        original_forward = self.model.forward
        trainer = self

        @wraps(original_forward)
        def aural_forward(model_self, *args, **kwargs):
            if args or "labels" not in kwargs:
                return original_forward(*args, **kwargs)
            return trainer._aural_forward_impl(kwargs)

        self.model._aural_original_forward = original_forward
        self.model.forward = MethodType(aural_forward, self.model)
        self.model._aural_forward_installed = True
        logger.info("[aural] installed top-level forward hook for DDP-visible custom loss")

    def _aural_forward_impl(self, inputs):
        inputs.pop("compute_loss_func", None)
        inputs.pop("loss_scale", None)
        inputs.pop("text_position_ids", None)
        inputs.pop("channel", None)
        inputs.pop("position_ids", None)
        output_router_logits = inputs.pop("output_router_logits", None)
        input_ids = inputs["input_ids"]
        labels = inputs.pop("labels")
        attention_mask = inputs.get("attention_mask")
        input_features = inputs.pop("input_features", None)
        feature_attention_mask = inputs.pop("feature_attention_mask", None)
        for mm_key in (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "video_second_per_grid",
            "audio_feature_lengths",
            "rope_deltas",
            "use_audio_in_video",
            "cache_position",
        ):
            inputs.pop(mm_key, None)
        if attention_mask is None:
            attention_mask = (input_ids != self.aural_pad_id).long()
        device = input_ids.device
        base_embeds = self._build_base_embeds(input_ids, input_features, feature_attention_mask)
        r = self._select_compression_r()
        if r == 1:
            embeds = base_embeds
            attn = attention_mask
            think_mask = self._build_think_mask(input_ids, attention_mask)
            gold_source = base_embeds
            ce_ids, ce_weights = self._hard_ce_targets(labels)
        else:
            embeds, attn, think_mask, gold_source, ce_ids, ce_weights = self._compress_batch(
                input_ids, base_embeds, labels, attention_mask, r
            )
        position_ids = torch.clamp_min(attn.long().cumsum(dim=1) - 1, 0)
        text_kwargs = {}
        if output_router_logits is not None:
            text_kwargs["output_router_logits"] = output_router_logits
        text_out = self._text_model(
            inputs_embeds=embeds,
            attention_mask=attn,
            position_ids=position_ids,
            use_cache=False,
            **text_kwargs,
        )
        hidden = text_out.last_hidden_state
        ce_loss, logits = self._causal_ce(hidden, ce_ids, ce_weights)
        embed_loss, entropy = self._latent_loss(hidden, gold_source, think_mask)
        want_pef = r > 1 and self.model.training and (self._pef_weight() > 0)
        if want_pef:
            pef_loss = self._pred_embed_forward(
                embeds=embeds,
                attn=attn,
                position_ids=position_ids,
                hidden=hidden,
                think_mask=think_mask,
                ce_ids=ce_ids,
                ce_weights=ce_weights,
                text_kwargs=text_kwargs,
            )
        else:
            pef_loss = torch.zeros((), dtype=ce_loss.dtype, device=ce_loss.device)
        loss_dev = ce_loss.device
        total = self.aural_ce_weight * ce_loss
        if self.aural_embed_weight != 0:
            total = total + self.aural_embed_weight * embed_loss.to(loss_dev)
        if self.aural_entropy_weight != 0:
            total = total + self.aural_entropy_weight * entropy.to(loss_dev)
        pef_weight = self._pef_weight()
        if pef_weight != 0:
            total = total + pef_weight * pef_loss.to(loss_dev)
        total = self._attach_conditional_parameter_anchors(total)
        outputs = {
            "loss": total.to(device),
            "logits": logits,
            "ce_loss": ce_loss,
            "embed_loss": embed_loss,
            "embed_nll": embed_loss,
            "pef_loss": pef_loss,
            "entropy": entropy,
            "r": torch.tensor(float(r), device=loss_dev),
        }
        return outputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        mode = "train" if self.model.training else "eval"
        self.custom_metrics[mode]["ce_loss"].update(outputs["ce_loss"].detach())
        self.custom_metrics[mode]["embed_loss"].update(outputs["embed_loss"].detach())
        self.custom_metrics[mode]["embed_nll"].update(outputs["embed_nll"].detach())
        self.custom_metrics[mode]["lss_loss"].update(outputs["pef_loss"].detach())
        if self.aural_entropy_weight != 0:
            self.custom_metrics[mode]["entropy"].update(outputs["entropy"].detach())
        self.custom_metrics[mode]["r"].update(outputs["r"].detach())
        if return_outputs:
            return (outputs["loss"], outputs)
        return outputs["loss"]

    def _chunk_targets(
        self,
        gold_source: torch.Tensor,
        think_mask: torch.Tensor,
        index_b: torch.Tensor,
        index_l: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.aural_chunk
        batch, length, feature = gold_source.shape
        padded_gold = torch.cat(
            [gold_source, gold_source.new_zeros(batch, chunk + 1, feature)], dim=1
        )
        padded_mask = torch.cat([think_mask, think_mask.new_zeros(batch, chunk)], dim=1)
        offsets = torch.arange(chunk, device=index_l.device)
        target_positions = index_l[:, None] + 1 + offsets
        anchor_positions = index_l[:, None] + offsets
        rows = index_b[:, None].expand_as(target_positions)
        targets = padded_gold[rows, target_positions]
        slot_mask = padded_mask[rows, anchor_positions]
        return (targets, slot_mask)

    def _strided_anchor_mask(self, think_mask: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(think_mask)
        for b in range(think_mask.shape[0]):
            positions = torch.nonzero(think_mask[b] > 0, as_tuple=False).flatten()
            if positions.numel():
                mask[b, positions[:: self.aural_chunk]] = 1.0
        return mask

    def _chunk_scatter_indices(self, think_mask: torch.Tensor):
        anchor_mask = self._strided_anchor_mask(think_mask)
        input_mask = self._latent_input_mask_from_think_mask(think_mask)
        index_b, index_l = torch.nonzero(anchor_mask > 0, as_tuple=True)
        offsets = torch.arange(self.aural_chunk, device=think_mask.device)
        target_l = index_l[:, None] + 1 + offsets
        target_b = index_b[:, None].expand_as(target_l)
        length = think_mask.shape[1]
        valid = target_l < length
        valid = valid & (input_mask[target_b, target_l.clamp(max=length - 1)] > 0)
        return (index_b, index_l, target_b, target_l, valid)

    def _pef_weight(self) -> float:
        weight = self.aural_pef_weight
        if weight != 0 and self.aural_pef_warmup_steps > 0:
            step = int(getattr(self.state, "global_step", 0) or 0)
            weight = weight * min(1.0, step / float(self.aural_pef_warmup_steps))
        return weight

    def _pred_embed_forward(
        self,
        embeds: torch.Tensor,
        attn: torch.Tensor,
        position_ids: torch.Tensor,
        hidden: torch.Tensor,
        think_mask: torch.Tensor,
        ce_ids: torch.Tensor,
        ce_weights: torch.Tensor,
        text_kwargs: dict,
    ) -> torch.Tensor:
        latent_input_mask = self._latent_input_mask_from_think_mask(think_mask)
        if latent_input_mask.sum() <= 0:
            return torch.zeros((), dtype=embeds.dtype, device=embeds.device)
        policy_parameter = next(self.latent_policy.parameters())
        policy_device, policy_dtype = (policy_parameter.device, policy_parameter.dtype)
        index_b, index_l, target_b, target_l, valid = self._chunk_scatter_indices(
            think_mask.to(policy_device)
        )
        if index_b.numel() == 0:
            return torch.zeros((), dtype=embeds.dtype, device=embeds.device)
        anchor_hidden = hidden.to(device=policy_device, dtype=policy_dtype)[index_b, index_l]
        mixture = self.latent_policy(anchor_hidden)
        prediction = mixture.rsample(temperature=self.aural_pef_temperature)
        input_mask = latent_input_mask.to(policy_device)
        scattered = torch.zeros(
            (*embeds.shape[:2], prediction.shape[-1]), device=policy_device, dtype=prediction.dtype
        )
        scattered = scattered.index_put((target_b[valid], target_l[valid]), prediction[valid])
        pred_next = scattered.to(device=embeds.device, dtype=embeds.dtype) * float(self.embeds_std)
        replace = input_mask.to(device=embeds.device, dtype=embeds.dtype).unsqueeze(-1)
        second_embeds = embeds * (1.0 - replace) + pred_next * replace
        second_out = self._text_model(
            inputs_embeds=second_embeds,
            attention_mask=attn,
            position_ids=position_ids,
            use_cache=False,
            **text_kwargs,
        )
        pef_loss, _ = self._causal_ce(second_out.last_hidden_state, ce_ids, ce_weights)
        return pef_loss

    def _latent_input_mask_from_think_mask(self, think_mask: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(think_mask)
        for b in range(think_mask.shape[0]):
            idx = torch.nonzero(think_mask[b] > 0, as_tuple=False).flatten()
            if idx.numel() > 1:
                mask[b, idx[1:]] = 1.0
        return mask

    def _hard_ce_targets(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        slots = max(1, self.aural_fixed_r if self.aural_fixed_r > 0 else self.aural_max_r)
        supervised = labels.ne(-100)
        ids = torch.zeros((*labels.shape, slots), dtype=torch.long, device=labels.device)
        weights = torch.zeros((*labels.shape, slots), dtype=torch.float32, device=labels.device)
        ids[..., 0] = labels.clamp_min(0)
        weights[..., 0] = supervised.to(weights.dtype)
        return (ids, weights)

    def _causal_ce(
        self, hidden: torch.Tensor, ce_ids: torch.Tensor, ce_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift_hidden = hidden[:, :-1, :]
        shift_ids = ce_ids[:, 1:].to(shift_hidden.device)
        shift_weights = ce_weights[:, 1:].to(shift_hidden.device)
        supervised = shift_weights.sum(dim=-1) > 0
        selected_hidden = shift_hidden[supervised]
        selected_ids = shift_ids[supervised]
        selected_weights = shift_weights[supervised]
        if selected_hidden.shape[0] == 0:
            raise ValueError("AURAL batch has no supervised causal-LM labels")
        logits = self._lm_head(selected_hidden)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        picked = log_probs.gather(dim=-1, index=selected_ids)
        loss = -(picked * selected_weights).sum(dim=-1).mean()
        if not getattr(self, "_aural_masked_ce_logged", False):
            logger.info(
                "[aural] supervised-only CE projection enabled: positions=%s/%s soft_latent_labels=%s",
                selected_hidden.shape[0],
                shift_ids.shape[0] * shift_ids.shape[1],
                self.aural_latent_ce_soft,
            )
            self._aural_masked_ce_logged = True
        return (loss, logits)

    def _latent_loss(
        self, hidden: torch.Tensor, gold_source: torch.Tensor, think_mask: torch.Tensor
    ):
        policy_parameter = next(self.latent_policy.parameters())
        policy_device = policy_parameter.device
        policy_dtype = policy_parameter.dtype
        hidden = hidden.to(device=policy_device, dtype=policy_dtype)
        gold_source = gold_source.to(policy_device)
        mask = think_mask.to(policy_device)
        index_b, index_l = torch.nonzero(mask > 0, as_tuple=True)
        empty = index_b.numel() == 0
        if empty:
            index_b = torch.zeros(1, dtype=torch.long, device=policy_device)
            index_l = torch.zeros(1, dtype=torch.long, device=policy_device)
        targets, slot_mask = self._chunk_targets(gold_source, mask, index_b, index_l)
        targets = targets.float() / self.embeds_std
        if empty:
            slot_mask = torch.ones_like(slot_mask)
        mixture = self.latent_policy(hidden[index_b, index_l])
        log_prob = mixture.log_prob(targets.detach(), slot_mask=slot_mask)
        valid_dimensions = (slot_mask.sum(dim=-1) * float(self.aural_hidden_size)).clamp_min(1.0)
        embed_loss = (-log_prob / valid_dimensions).mean()
        entropy = (
            mixture.entropy_upper_bound() / float(self.aural_chunk * self.aural_hidden_size)
        ).mean()
        if empty:
            embed_loss = embed_loss * 0.0
            entropy = entropy * 0.0
        return (embed_loss, entropy)
