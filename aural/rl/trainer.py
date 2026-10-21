from __future__ import annotations

import json
import math
from typing import Any, List, Sequence

import torch
from swift.utils import get_logger
from torch.utils.checkpoint import checkpoint

from ..sft import SFTTrainer
from .config import RLConfig
from .experience import ActionLogProbs, LatentRolloutGroup
from .rewards import RewardScorer

logger = get_logger()


def _log1mexp(log_probability: torch.Tensor) -> torch.Tensor:
    value = log_probability.float().clamp(max=-1e-07)
    cutoff = -math.log(2.0)
    return torch.where(
        value < cutoff, torch.log1p(-torch.exp(value)), torch.log(-torch.expm1(value))
    )


def _one(value: Any, name: str) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must contain one value, got shape={tuple(value.shape)}")
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{name} must contain one value, got {len(value)}")
        return value[0]
    return value


def _mean_tensors(values: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    nonempty = [value.float().mean() for value in values if value.numel()]
    if not nonempty:
        return torch.zeros((), device=device)
    return torch.stack(nonempty).mean()


class RLTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rl_config = RLConfig.from_env()
        if self.args.per_device_train_batch_size != 1:
            raise ValueError("RL uses one prompt per device; group_size controls the rollouts")
        self._scorer = RewardScorer()
        self._last_rollout_microbatch = self.rl_config.group_size
        self._model_context_length = self._thinker.config.text_config.max_position_embeddings

    def _extract_rollout_input(self, inputs):
        if "aural_sample" not in inputs:
            raise ValueError("Missing RL sample metadata")
        sample = json.loads(_one(inputs.pop("aural_sample"), "aural_sample"))
        input_ids = inputs["input_ids"]
        if input_ids.shape[0] != 1:
            raise ValueError("RL expects one prompt per device")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.aural_pad_id).long()
        valid_length = int(attention_mask[0].sum().item())
        _, reason_start, _, _ = self._find_reason_span(input_ids[0].tolist(), valid_length)
        budget = (
            reason_start + self.rl_config.max_latent_forward + 1 + self.rl_config.max_answer_tokens
        )
        if budget > self._model_context_length:
            raise ValueError("Prompt and rollout budget exceed the model context length")
        prompt_inputs = (
            input_ids[:, :reason_start],
            inputs.get("input_features"),
            inputs.get("feature_attention_mask"),
        )
        with torch.no_grad():
            prompt = self._build_base_embeds(*prompt_inputs).detach()
        return (prompt, sample, input_ids.device, prompt_inputs)

    def _aural_forward_impl(self, inputs):
        prompt, sample, device, prompt_inputs = self._extract_rollout_input(inputs)
        rollout = self._rollout(prompt)
        quality = torch.tensor(
            self._scorer(sample, rollout.answers), device=device, dtype=torch.float32
        )
        lengths = rollout.latent_lengths.float()
        shortness = 1 - (lengths - 1) / max(1, self.rl_config.max_latent_forward - 1)
        rewards = quality * (1 + self.rl_config.length_reward_weight * shortness)
        rewards -= (~rollout.hit_eol).float() * self.rl_config.missing_eol_penalty
        rewards -= (~rollout.hit_eos).float() * self.rl_config.missing_eos_penalty
        mixed = bool(
            (quality <= self.rl_config.mixed_low).any()
            and (quality >= self.rl_config.mixed_high).any()
        )
        rollout.prompt_embeds = self._build_base_embeds(*prompt_inputs)
        current = self._score_actions(rollout, self.latent_policy)
        old = ActionLogProbs(
            rollout.old_latent_logps, rollout.old_stop_logps, rollout.old_answer_logps
        )
        delta = self._flatten_delta(current, old)
        if not torch.isfinite(delta).all():
            raise FloatingPointError("Non-finite action log probabilities")
        advantages = (rewards - rewards.mean()) / (rewards.std() + self.rl_config.advantage_epsilon)
        if self.rl_config.require_mixed_group and (not mixed):
            advantages = torch.zeros_like(advantages)
        latent_loss = self._ppo_loss(current.latent, old.latent, advantages)
        stop_loss = self._ppo_loss(current.stop, old.stop, advantages)
        answer_loss = self._ppo_loss(current.answer, old.answer, advantages)
        loss = (
            self.rl_config.latent_loss_weight * latent_loss
            + self.rl_config.stop_loss_weight * stop_loss
            + self.rl_config.answer_loss_weight * answer_loss
        )
        loss = self._attach_conditional_parameter_anchors(loss)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite GRPO loss")
        return {
            "loss": loss.to(device),
            "quality": quality.mean(),
            "reward": rewards.mean(),
            "mixed_group": quality.new_tensor(float(mixed)),
            "advantage_std": advantages.std(unbiased=False).detach(),
            "latent_tokens": lengths.mean(),
            "hit_eol": rollout.hit_eol.float().mean(),
            "hit_eos": rollout.hit_eos.float().mean(),
            "latent_policy_loss": latent_loss.detach(),
            "stop_policy_loss": stop_loss.detach(),
            "answer_policy_loss": answer_loss.detach(),
            "old_logprob_delta": delta.abs().mean().detach(),
        }

    @staticmethod
    def _position_ids(attention_mask: torch.Tensor, width: int | None = None) -> torch.Tensor:
        position_ids = torch.clamp_min(attention_mask.long().cumsum(dim=1) - 1, 0)
        return position_ids if width is None else position_ids[:, -width:]

    def _append_with_cache(
        self,
        embeddings: torch.Tensor,
        new_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values,
    ):
        width = embeddings.shape[1]
        attention_mask = torch.cat([attention_mask, new_mask.to(attention_mask.dtype)], dim=1)
        output = self._text_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=self._position_ids(attention_mask, width),
            past_key_values=past_key_values,
            use_cache=True,
        )
        return (output, attention_mask)

    @staticmethod
    def _sample_tokens(
        logits: torch.Tensor, *, temperature: float, top_p: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_log_probs = torch.log_softmax(logits.float(), dim=-1)
        probabilities = torch.softmax(logits.float() / temperature, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative - sorted_probs >= top_p
            sorted_probs = sorted_probs.masked_fill(remove, 0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            sampled_sorted = torch.multinomial(sorted_probs, 1)
            token_ids = sorted_indices.gather(-1, sampled_sorted).squeeze(-1)
        else:
            token_ids = torch.multinomial(probabilities, 1).squeeze(-1)
        selected_log_probs = raw_log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        return (token_ids, selected_log_probs)

    def _sample_eol_events(
        self, logits: torch.Tensor
    ) -> tuple[List[int], List[torch.Tensor], List[bool]]:
        raw_log_probs = torch.log_softmax(logits.float(), dim=-1)
        raw_stop = raw_log_probs[..., self.think_close_id]
        raw_continue = _log1mexp(raw_stop)
        sample_probs = torch.softmax(logits.float() / self.rl_config.eol_temperature, dim=-1)
        sample_stop = sample_probs[..., self.think_close_id]
        draws = torch.rand_like(sample_stop) < sample_stop
        accepted: List[int] = []
        event_logps: List[torch.Tensor] = []
        stopped: List[bool] = []
        for row in range(logits.shape[0]):
            count = logits.shape[1]
            did_stop = False
            values = []
            for slot in range(logits.shape[1]):
                event = bool(draws[row, slot].item())
                values.append(raw_stop[row, slot] if event else raw_continue[row, slot])
                if event:
                    count = slot + 1
                    did_stop = True
                    break
            accepted.append(count)
            event_logps.append(torch.stack(values))
            stopped.append(did_stop)
        return (accepted, event_logps, stopped)

    def _rollout_impl(
        self, prompt_embeds: torch.Tensor, group_size: int | None
    ) -> LatentRolloutGroup:
        config = self.rl_config
        group = config.group_size if group_size is None else group_size
        if not 1 <= group <= config.group_size:
            raise ValueError(f"rollout microbatch must be in [1, {config.group_size}], got {group}")
        device = prompt_embeds.device
        dtype = prompt_embeds.dtype
        hidden_size = prompt_embeds.shape[-1]
        prompt = prompt_embeds.expand(group, -1, -1).contiguous()
        attention_mask = torch.ones(prompt.shape[:2], device=device, dtype=torch.long)
        output = self._text_model(
            inputs_embeds=prompt,
            attention_mask=attention_mask,
            position_ids=self._position_ids(attention_mask),
            use_cache=True,
        )
        past_key_values = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1, :]
        latent_parts: List[List[torch.Tensor]] = [[] for _ in range(group)]
        chunk_lengths: List[List[int]] = [[] for _ in range(group)]
        old_latent: List[List[torch.Tensor]] = [[] for _ in range(group)]
        old_stop: List[List[torch.Tensor]] = [[] for _ in range(group)]
        hit_eol = torch.zeros(group, device=device, dtype=torch.bool)
        active = torch.ones(group, device=device, dtype=torch.bool)
        latent_counts = torch.zeros(group, device=device, dtype=torch.long)
        while bool(active.any().item()):
            active_indices = torch.nonzero(active, as_tuple=False).flatten()
            policy_parameter = next(self.latent_policy.parameters())
            mixture = self.latent_policy(
                last_hidden[active_indices].to(
                    device=policy_parameter.device, dtype=policy_parameter.dtype
                )
            )
            sampled = mixture.rsample(temperature=config.latent_temperature).detach()
            chunk = torch.zeros((group, self.aural_chunk, hidden_size), device=device, dtype=dtype)
            chunk[active_indices] = sampled.to(device=device, dtype=dtype) * float(self.embeds_std)
            provisional_mask = active[:, None].expand(-1, self.aural_chunk).clone()
            output, attention_mask = self._append_with_cache(
                chunk, provisional_mask, attention_mask, past_key_values
            )
            past_key_values = output.past_key_values
            chunk_hidden = output.last_hidden_state
            active_logits = self._lm_head(
                chunk_hidden[active_indices].to(self._lm_head.weight.device)
            )
            accepted, event_logps, stopped = self._sample_eol_events(active_logits)
            accepted_mask = torch.zeros((group, self.aural_chunk), device=device, dtype=torch.long)
            slot_masks = torch.zeros(
                (active_indices.numel(), self.aural_chunk),
                device=sampled.device,
                dtype=torch.float32,
            )
            for local_idx, global_idx_tensor in enumerate(active_indices):
                global_idx = int(global_idx_tensor.item())
                remaining = config.max_latent_forward - int(latent_counts[global_idx].item())
                sampled_count = accepted[local_idx]
                count = min(sampled_count, remaining)
                accepted_mask[global_idx, :count] = 1
                slot_masks[local_idx, :count] = 1
                latent_parts[global_idx].append(sampled[local_idx, :count].detach())
                chunk_lengths[global_idx].append(count)
                old_stop[global_idx].append(event_logps[local_idx][:count].detach())
                latent_counts[global_idx] += count
                stopped_within_cap = stopped[local_idx] and sampled_count <= remaining
                if stopped_within_cap:
                    hit_eol[global_idx] = True
                    active[global_idx] = False
                elif int(latent_counts[global_idx].item()) >= config.max_latent_forward:
                    active[global_idx] = False
            attention_mask[:, -self.aural_chunk :] = accepted_mask
            normalized_logp = mixture.log_prob(sampled, slot_mask=slot_masks)
            dimensions = slot_masks.sum(dim=-1) * float(hidden_size)
            normalized_logp = normalized_logp / dimensions.clamp_min(1.0)
            for local_idx, global_idx_tensor in enumerate(active_indices):
                old_latent[int(global_idx_tensor.item())].append(
                    normalized_logp[local_idx].detach()
                )
            last_hidden = chunk_hidden[:, -1, :]
        separator_ids = torch.full(
            (group, 1),
            self.think_close_id,
            device=self._embed_tokens.weight.device,
            dtype=torch.long,
        )
        separator_embeds = self._embed_tokens(separator_ids).to(device=device, dtype=dtype)
        output, attention_mask = self._append_with_cache(
            separator_embeds,
            torch.ones((group, 1), device=device, dtype=torch.long),
            attention_mask,
            past_key_values,
        )
        past_key_values = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1, :]
        eos_id = int(self.tokenizer.eos_token_id)
        pad_id = int(self.tokenizer.pad_token_id)
        answer_ids: List[List[int]] = [[] for _ in range(group)]
        old_answer: List[List[torch.Tensor]] = [[] for _ in range(group)]
        answer_active = torch.ones(group, device=device, dtype=torch.bool)
        hit_eos = torch.zeros(group, device=device, dtype=torch.bool)
        for _ in range(config.max_answer_tokens):
            active_indices = torch.nonzero(answer_active, as_tuple=False).flatten()
            if active_indices.numel() == 0:
                break
            logits = self._lm_head(last_hidden[active_indices].to(self._lm_head.weight.device))
            tokens, logps = self._sample_tokens(
                logits, temperature=config.answer_temperature, top_p=config.answer_top_p
            )
            full_tokens = torch.full((group,), pad_id, device=device, dtype=torch.long)
            full_mask = torch.zeros((group, 1), device=device, dtype=torch.long)
            for local_idx, global_idx_tensor in enumerate(active_indices):
                global_idx = int(global_idx_tensor.item())
                token_id = int(tokens[local_idx].item())
                full_tokens[global_idx] = token_id
                full_mask[global_idx, 0] = 1
                answer_ids[global_idx].append(token_id)
                old_answer[global_idx].append(logps[local_idx].detach())
                if token_id == eos_id:
                    hit_eos[global_idx] = True
                    answer_active[global_idx] = False
            if not bool(answer_active.any().item()):
                break
            token_embeds = self._embed_tokens(
                full_tokens.to(device=self._embed_tokens.weight.device).unsqueeze(1)
            ).to(device=device, dtype=dtype)
            output, attention_mask = self._append_with_cache(
                token_embeds, full_mask, attention_mask, past_key_values
            )
            past_key_values = output.past_key_values
            last_hidden = output.last_hidden_state[:, -1, :]
        latent_values = [
            torch.cat(parts, dim=0)
            if parts
            else torch.empty((0, hidden_size), device=device, dtype=dtype)
            for parts in latent_parts
        ]
        answer_tensors = [torch.tensor(ids, device=device, dtype=torch.long) for ids in answer_ids]
        answers = []
        for ids in answer_ids:
            visible = [token_id for token_id in ids if token_id != eos_id]
            answers.append(
                self.tokenizer.decode(
                    visible, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            )
        return LatentRolloutGroup(
            prompt_embeds=prompt_embeds.detach(),
            latent_values=latent_values,
            chunk_lengths=chunk_lengths,
            answer_ids=answer_tensors,
            answers=answers,
            old_latent_logps=[torch.stack(values) for values in old_latent],
            old_stop_logps=[torch.cat(values) for values in old_stop],
            old_answer_logps=[torch.stack(values) for values in old_answer],
            hit_eol=hit_eol,
            hit_eos=hit_eos,
        )

    @torch.no_grad()
    def _rollout(self, prompt_embeds: torch.Tensor) -> LatentRolloutGroup:
        was_training = self.model.training
        self.model.eval()
        try:
            max_sequence_length = (
                prompt_embeds.shape[1]
                + self.rl_config.max_latent_forward
                + 1
                + self.rl_config.max_answer_tokens
            )
            microbatch = max(
                1,
                min(
                    self.rl_config.group_size,
                    self.rl_config.rollout_token_budget // max_sequence_length,
                ),
            )
            self._last_rollout_microbatch = microbatch
            groups = []
            remaining = self.rl_config.group_size
            while remaining:
                current = min(microbatch, remaining)
                groups.append(self._rollout_impl(prompt_embeds, group_size=current))
                remaining -= current
            if len(groups) == 1:
                return groups[0]
            return LatentRolloutGroup(
                prompt_embeds=prompt_embeds.detach(),
                latent_values=[value for group in groups for value in group.latent_values],
                chunk_lengths=[value for group in groups for value in group.chunk_lengths],
                answer_ids=[value for group in groups for value in group.answer_ids],
                answers=[value for group in groups for value in group.answers],
                old_latent_logps=[value for group in groups for value in group.old_latent_logps],
                old_stop_logps=[value for group in groups for value in group.old_stop_logps],
                old_answer_logps=[value for group in groups for value in group.old_answer_logps],
                hit_eol=torch.cat([group.hit_eol for group in groups]),
                hit_eos=torch.cat([group.hit_eos for group in groups]),
            )
        finally:
            if was_training:
                self.model.train()

    def _score_stop_hidden(self, hidden: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        results = []
        for start in range(0, hidden.shape[0], self.rl_config.logit_chunk_size):
            chunk = hidden[start : start + self.rl_config.logit_chunk_size]
            logits = self._lm_head(chunk.to(self._lm_head.weight.device))
            log_stop = torch.log_softmax(logits.float(), dim=-1)[:, self.think_close_id]
            log_continue = _log1mexp(log_stop)
            current_events = events[start : start + chunk.shape[0]].to(log_stop.device)
            results.append(torch.where(current_events, log_stop, log_continue))
        return torch.cat(results)

    def _score_answer_hidden(self, hidden: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        results = []
        for start in range(0, hidden.shape[0], self.rl_config.logit_chunk_size):
            chunk = hidden[start : start + self.rl_config.logit_chunk_size]
            logits = self._lm_head(chunk.to(self._lm_head.weight.device))
            ids = token_ids[start : start + chunk.shape[0]].to(logits.device)
            results.append(
                torch.log_softmax(logits.float(), dim=-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
            )
        return torch.cat(results)

    def _score_actions(self, rollout: LatentRolloutGroup, latent_policy) -> ActionLogProbs:
        prompt_length = rollout.prompt_embeds.shape[1]
        prompt = rollout.prompt_embeds[0]
        device = prompt.device
        dtype = prompt.dtype
        policy_parameter = next(latent_policy.parameters())
        separator = self._embed_tokens(
            torch.tensor(
                [self.think_close_id], device=self._embed_tokens.weight.device, dtype=torch.long
            )
        ).to(device=device, dtype=dtype)
        latent_scores: List[torch.Tensor] = []
        stop_scores: List[torch.Tensor] = []
        answer_scores: List[torch.Tensor] = []
        for sample_idx in range(rollout.group_size):
            latent = rollout.latent_values[sample_idx].to(device=device, dtype=dtype)
            latent_count = rollout.latent_values[sample_idx].shape[0]
            close_index = prompt_length + latent_count
            answer_count = rollout.answer_ids[sample_idx].numel()
            answer = self._embed_tokens(
                rollout.answer_ids[sample_idx].to(self._embed_tokens.weight.device)
            ).to(device=device, dtype=dtype)
            anchor_positions = []
            offset = 0
            for count in rollout.chunk_lengths[sample_idx]:
                anchor_positions.append(prompt_length - 1 + offset)
                offset += count
            if offset != latent_count:
                raise RuntimeError(
                    f"latent chunk accounting mismatch for rollout {sample_idx}: chunks={offset}, values={latent_count}"
                )
            selected_positions = torch.tensor(
                anchor_positions
                + list(range(prompt_length, prompt_length + latent_count))
                + list(range(close_index, close_index + answer_count)),
                device=device,
                dtype=torch.long,
            )

            def selected_hidden_forward(
                prompt_value: torch.Tensor,
                latent_value: torch.Tensor,
                separator_value: torch.Tensor,
                answer_value: torch.Tensor,
                selected_positions_value: torch.Tensor,
            ) -> torch.Tensor:
                sequence = torch.cat(
                    [
                        prompt_value,
                        latent_value * float(self.embeds_std),
                        separator_value,
                        answer_value,
                    ],
                    dim=0,
                ).unsqueeze(0)
                attention_mask = torch.ones(
                    sequence.shape[:2], device=sequence.device, dtype=torch.long
                )
                output = self._text_model(
                    inputs_embeds=sequence,
                    attention_mask=attention_mask,
                    position_ids=self._position_ids(attention_mask),
                    use_cache=False,
                )
                return output.last_hidden_state[0].index_select(0, selected_positions_value)

            selected_hidden = checkpoint(
                selected_hidden_forward,
                prompt,
                latent,
                separator,
                answer,
                selected_positions,
                use_reentrant=False,
            )
            anchor_count = len(anchor_positions)
            anchor_hidden = selected_hidden[:anchor_count]
            stop_hidden = selected_hidden[anchor_count : anchor_count + latent_count]
            answer_hidden = selected_hidden[anchor_count + latent_count :]
            targets = []
            slot_masks = []
            offset = 0
            for count in rollout.chunk_lengths[sample_idx]:
                target = torch.zeros(
                    (self.aural_chunk, self.aural_hidden_size),
                    device=latent.device,
                    dtype=latent.dtype,
                )
                target[:count] = latent[offset : offset + count]
                mask = torch.zeros(self.aural_chunk, device=latent.device, dtype=torch.float32)
                mask[:count] = 1
                targets.append(target)
                slot_masks.append(mask)
                offset += count
            mixture = latent_policy(
                anchor_hidden.to(device=policy_parameter.device, dtype=policy_parameter.dtype)
            )
            target_tensor = torch.stack(targets).to(policy_parameter.device)
            slot_tensor = torch.stack(slot_masks).to(policy_parameter.device)
            latent_logps = mixture.log_prob(target_tensor, slot_mask=slot_tensor)
            latent_scores.append(
                latent_logps
                / (slot_tensor.sum(dim=-1) * float(self.aural_hidden_size)).clamp_min(1.0)
            )
            events = torch.zeros(latent_count, device=stop_hidden.device, dtype=torch.bool)
            if bool(rollout.hit_eol[sample_idx].item()):
                events[-1] = True
            stop_scores.append(self._score_stop_hidden(stop_hidden, events))
            answer_scores.append(
                self._score_answer_hidden(answer_hidden, rollout.answer_ids[sample_idx])
            )
        return ActionLogProbs(latent=latent_scores, stop=stop_scores, answer=answer_scores)

    @staticmethod
    def _flatten_delta(current: ActionLogProbs, old: ActionLogProbs) -> torch.Tensor:
        values = []
        for current_group, old_group in (
            (current.latent, old.latent),
            (current.stop, old.stop),
            (current.answer, old.answer),
        ):
            values.extend(
                (
                    (new.float() - previous.float()).reshape(-1)
                    for new, previous in zip(current_group, old_group)
                )
            )
        return torch.cat(values)

    def _ppo_loss(
        self, current: Sequence[torch.Tensor], old: Sequence[torch.Tensor], advantages: torch.Tensor
    ) -> torch.Tensor:
        losses = []
        for sample_idx, (new, previous) in enumerate(zip(current, old)):
            ratio = torch.exp((new.float() - previous.float()).clamp(-20, 20))
            advantage = advantages[sample_idx].to(ratio.device)
            unclipped = ratio * advantage
            clipped = (
                ratio.clamp(1.0 - self.rl_config.clip_epsilon, 1.0 + self.rl_config.clip_epsilon)
                * advantage
            )
            losses.append(-torch.minimum(unclipped, clipped).mean())
        return torch.stack(losses).mean()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        mode = "train" if self.model.training else "eval"
        for key, value in outputs.items():
            if key != "loss" and isinstance(value, torch.Tensor) and (value.numel() == 1):
                self.custom_metrics[mode][key].update(value.detach())
        if return_outputs:
            return (outputs["loss"], outputs)
        return outputs["loss"]
