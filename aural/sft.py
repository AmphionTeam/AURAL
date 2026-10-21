import os
from pathlib import Path
from typing import List

import torch
from safetensors.torch import load_file, save_file
from swift.utils import get_logger

from .audio import merge_audio_embeds
from .config import AuralConfig
from .markers import encode_think_markers, find_reason_span
from .sft_base import LatentSFTBase

logger = get_logger()


class SFTTrainer(LatentSFTBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.think_open_ids, self.think_close_ids = encode_think_markers(self.tokenizer)
        self._audio_encoder_trainable = any(
            parameter.requires_grad
            for name, parameter in self._thinker.named_parameters()
            if name.startswith("audio_tower.")
        )
        self._conditional_gradient_anchors = tuple(
            (name, parameter)
            for name, parameter in self._thinker.named_parameters()
            if parameter.requires_grad and name.startswith("audio_tower.")
        )
        self.latent_policy.to(dtype=self._embed_tokens.weight.dtype)
        source = os.environ.get("AURAL_INIT_CHECKPOINT", "")
        if source and Path(source, "aural_config.json").is_file():
            config = AuralConfig.load(source)
            expected = {
                "compression_factor": self.aural_fixed_r,
                "chunk_size": self.aural_chunk,
                "hidden_size": self.aural_hidden_size,
                "gmm_components": self.aural_gmm_components,
                "lowrank": self.aural_lowrank,
            }
            for key, value in expected.items():
                if getattr(config, key) != value:
                    raise ValueError(f"Checkpoint {key} differs from the training configuration")
            self.latent_policy.load_state_dict(
                load_file(str(Path(source, "latent_policy.safetensors"))), strict=True
            )
            self.embeds_std = config.embeds_std

    def _select_compression_r(self):
        return self.aural_fixed_r

    def _latent_ce_slots(self, grouped_ids, grouped_valid):
        weights = grouped_valid / grouped_valid.sum(dim=1, keepdim=True).clamp_min(1)
        return grouped_ids, weights.float()

    def _save_model(self, output_dir=None, state_dict=None):
        super()._save_model(output_dir=output_dir, state_dict=state_dict)
        if output_dir is None or not self.is_world_process_zero():
            return
        state = {
            key: value.detach().cpu().contiguous().clone()
            for key, value in self.latent_policy.state_dict().items()
        }
        save_file(state, str(Path(output_dir, "latent_policy.safetensors")))
        config = AuralConfig(
            compression_factor=self.aural_fixed_r,
            chunk_size=self.aural_chunk,
            gmm_components=self.aural_gmm_components,
            lowrank=self.aural_lowrank,
            mixer_mult=self.aural_mixer_mult,
            hidden_size=self.aural_hidden_size,
            lp_intermediate=self.aural_lp_intermediate,
            embeds_std=self.embeds_std,
            gmm_min_log_std=self.aural_gmm_min_log_std,
            gmm_max_log_std=self.aural_gmm_max_log_std,
            lss_warmup_steps=self.aural_pef_warmup_steps,
            stage="rl" if hasattr(self, "rl_config") else "sft",
        )
        config.save(output_dir)

    def _find_reason_span(self, row, valid_len: int):
        return find_reason_span(
            row, self.think_open_ids, self.think_close_ids, valid_len=valid_len, require_unique=True
        )

    def _build_base_embeds(self, input_ids, input_features, feature_attention_mask):
        base_embeds = self._embed_tokens(input_ids)
        return merge_audio_embeds(
            thinker=self._thinker,
            input_ids=input_ids,
            inputs_embeds=base_embeds,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_token_id=self._audio_token_id,
            ensure_audio_autograd=self._audio_encoder_trainable,
        )

    def _attach_conditional_parameter_anchors(self, loss: torch.Tensor) -> torch.Tensor:
        anchor = torch.zeros((), dtype=loss.dtype, device=loss.device)
        for _name, parameter in self._conditional_gradient_anchors:
            anchor = anchor + parameter.reshape(-1)[0].to(loss.dtype) * 0
        return loss + anchor

    def _build_think_mask(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        mask = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=input_ids.device)
        rows = input_ids.tolist()
        for batch_idx, row in enumerate(rows):
            valid_len = int(attention_mask[batch_idx].sum().item())
            open_start, _, close_start, _ = self._find_reason_span(row, valid_len)
            mask[batch_idx, open_start:close_start] = 1.0
        return mask

    def _compress_batch(self, input_ids, base_embeds, labels, attention_mask, r):
        batch_size, _, hidden_size = base_embeds.shape
        device = base_embeds.device
        rows = input_ids.tolist()
        seqs_embed: List[torch.Tensor] = []
        seqs_think: List[torch.Tensor] = []
        seqs_gold: List[torch.Tensor] = []
        seqs_ce_ids: List[torch.Tensor] = []
        seqs_ce_weights: List[torch.Tensor] = []

        def hard_slots(label_row: torch.Tensor):
            length = label_row.shape[0]
            slot_ids = torch.zeros((length, r), dtype=torch.long, device=device)
            slot_weights = torch.zeros((length, r), dtype=torch.float32, device=device)
            slot_ids[:, 0] = label_row.clamp_min(0)
            slot_weights[:, 0] = label_row.ne(-100).to(torch.float32)
            return (slot_ids, slot_weights)

        for batch_idx, row in enumerate(rows):
            valid_len = int(attention_mask[batch_idx].sum().item())
            open_start, reason_start, close_start, _ = self._find_reason_span(row, valid_len)
            if reason_start != open_start + 1:
                raise ValueError(
                    f"The original AURAL opening separator must be exactly one token: open_start={open_start}, reason_start={reason_start}."
                )
            emb_row = base_embeds[batch_idx]
            label_row = labels[batch_idx]
            prefix_embed = emb_row[:reason_start]
            prefix_label = label_row[:reason_start]
            reason_embed = emb_row[reason_start:close_start]
            reason_ids = input_ids[batch_idx, reason_start:close_start]
            answer_embed = emb_row[close_start:valid_len]
            answer_label = label_row[close_start:valid_len]
            n_reason = reason_embed.shape[0]
            if n_reason <= 0:
                raise ValueError(f"Qwen2.5 AURAL sample {batch_idx} has an empty reasoning span.")
            remainder = -n_reason % r
            if remainder:
                reason_embed_padded = torch.cat(
                    [reason_embed, reason_embed.new_zeros(remainder, hidden_size)], dim=0
                )
                valid = torch.cat(
                    [torch.ones(n_reason, device=device), torch.zeros(remainder, device=device)]
                )
                reason_ids_padded = torch.cat(
                    [reason_ids, reason_ids.new_full((remainder,), self.aural_pad_id)]
                )
            else:
                reason_embed_padded = reason_embed
                valid = torch.ones(n_reason, device=device)
                reason_ids_padded = reason_ids
            n_chunks = reason_embed_padded.shape[0] // r
            grouped_embed = reason_embed_padded.reshape(n_chunks, r, hidden_size)
            grouped_valid = valid.reshape(n_chunks, r)
            counts = grouped_valid.sum(dim=1)
            denominator = counts.sqrt() if self.aural_sqrt_mean else counts
            compressed = (grouped_embed * grouped_valid.unsqueeze(-1)).sum(dim=1) / (
                denominator.unsqueeze(-1) + 1e-05
            )
            grouped_ids = reason_ids_padded.reshape(n_chunks, r)
            compressed_ids, compressed_weights = self._latent_ce_slots(grouped_ids, grouped_valid)
            embed_sequence = torch.cat([prefix_embed, compressed, answer_embed], dim=0)
            latent_mask = torch.zeros(embed_sequence.shape[0], dtype=torch.float32, device=device)
            anchor = prefix_embed.shape[0] - 1
            latent_mask[anchor : prefix_embed.shape[0] + n_chunks] = 1.0
            prefix_ids, prefix_weights = hard_slots(prefix_label)
            answer_ids, answer_weights = hard_slots(answer_label)
            seqs_embed.append(embed_sequence)
            seqs_think.append(latent_mask)
            seqs_gold.append(embed_sequence)
            seqs_ce_ids.append(torch.cat([prefix_ids, compressed_ids, answer_ids], dim=0))
            seqs_ce_weights.append(
                torch.cat([prefix_weights, compressed_weights, answer_weights], dim=0)
            )
        max_len = max((sequence.shape[0] for sequence in seqs_embed))
        embeds = base_embeds.new_zeros(batch_size, max_len, hidden_size)
        compressed_attention_mask = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=device
        )
        think_mask = torch.zeros(batch_size, max_len, dtype=torch.float32, device=device)
        gold_source = base_embeds.new_zeros(batch_size, max_len, hidden_size)
        ce_ids = torch.zeros(batch_size, max_len, r, dtype=torch.long, device=device)
        ce_weights = torch.zeros(batch_size, max_len, r, dtype=torch.float32, device=device)
        for batch_idx, sequence in enumerate(seqs_embed):
            length = sequence.shape[0]
            embeds[batch_idx, :length] = sequence
            compressed_attention_mask[batch_idx, :length] = 1
            think_mask[batch_idx, :length] = seqs_think[batch_idx]
            gold_source[batch_idx, :length] = seqs_gold[batch_idx]
            ce_ids[batch_idx, :length] = seqs_ce_ids[batch_idx]
            ce_weights[batch_idx, :length] = seqs_ce_weights[batch_idx]
        return (embeds, compressed_attention_mask, think_mask, gold_source, ce_ids, ce_weights)

    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        final_checkpoint = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
        last_checkpoint = getattr(self.state, "last_model_checkpoint", None)
        if self.state.global_step > 0 and last_checkpoint != str(final_checkpoint):
            logger.info(
                "[qwen25-chunk] final step is outside the periodic save interval; saving checkpoint-%s now",
                self.state.global_step,
            )
            self._save_checkpoint(self.model, trial=None)
        return result
