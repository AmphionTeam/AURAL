from __future__ import annotations

from typing import Optional

import torch


def module_device(module) -> torch.device:
    for parameter in module.parameters(recurse=True):
        return parameter.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device("cpu")


def resolve_audio_token_id(thinker, tokenizer=None, model=None) -> int:
    for obj in (
        getattr(thinker, "config", None),
        getattr(getattr(model, "config", None), "thinker_config", None),
        getattr(model, "config", None),
    ):
        if obj is None:
            continue
        for name in ("audio_token_id", "audio_token_index"):
            value = getattr(obj, name, None)
            if value is not None:
                return int(value)
    if tokenizer is not None:
        ids = tokenizer.encode("<|AUDIO|>", add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    raise ValueError("Cannot resolve the Qwen2.5-Omni <|AUDIO|> token id.")


def merge_audio_embeds(
    *,
    thinker,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    input_features: Optional[torch.Tensor],
    feature_attention_mask: Optional[torch.Tensor],
    audio_token_id: int,
    ensure_audio_autograd: bool = False,
) -> torch.Tensor:
    text_only = input_features is None
    if text_only and (not ensure_audio_autograd):
        return inputs_embeds
    if not hasattr(thinker, "get_audio_features"):
        raise AttributeError("Qwen2.5-Omni thinker has no get_audio_features method.")
    audio_tower = thinker.audio_tower
    audio_device = module_device(audio_tower)
    audio_dtype = getattr(audio_tower, "dtype", inputs_embeds.dtype)
    if text_only:
        num_mel_bins = int(getattr(audio_tower, "num_mel_bins", 128))
        input_features = torch.zeros((1, num_mel_bins, 128), device=audio_device, dtype=audio_dtype)
        feature_attention_mask = torch.ones((1, 128), device=audio_device, dtype=torch.bool)
    elif feature_attention_mask is None:
        raise ValueError("Qwen2.5-Omni audio input requires feature_attention_mask.")
    input_features = input_features.to(device=audio_device, dtype=audio_dtype)
    feature_attention_mask = feature_attention_mask.to(device=audio_device)
    audio_features = thinker.get_audio_features(
        input_features, feature_attention_mask=feature_attention_mask
    )
    if hasattr(audio_features, "last_hidden_state"):
        audio_features = audio_features.last_hidden_state
    audio_features = audio_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    if text_only:
        return inputs_embeds + audio_features.mean() * 0
    audio_mask = (
        (input_ids.to(inputs_embeds.device) == int(audio_token_id))
        .unsqueeze(-1)
        .expand_as(inputs_embeds)
    )
    n_slots = int(audio_mask[..., 0].sum().item())
    if n_slots == 0:
        raise ValueError(
            "Audio features were provided, but input_ids contain no expanded <|AUDIO|> slots."
        )
    expected_numel = n_slots * inputs_embeds.shape[-1]
    if audio_features.numel() != expected_numel:
        raise ValueError(
            f"Qwen2.5-Omni audio features/token slots mismatch: slots={n_slots}, audio_features={tuple(audio_features.shape)}, hidden_size={inputs_embeds.shape[-1]}."
        )
    return inputs_embeds.masked_scatter(audio_mask, audio_features)
