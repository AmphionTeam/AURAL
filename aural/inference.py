from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from swift.infer_engine import InferRequest

from .audio import merge_audio_embeds, resolve_audio_token_id
from .latent_policy import ChunkLatentPolicy, ChunkMixtureOutput


def clean_answer(answer):
    answer = answer.strip()
    if answer.startswith("Answer:"):
        answer = answer[len("Answer:") :].lstrip()
    return answer


def load_generator(model="YuxiangW/AURAL", *, device="cuda:0", attn_impl="flash_attn", **overrides):
    import os
    from dataclasses import fields

    from safetensors.torch import load_file
    from swift.model import get_model_processor
    from swift.template import get_template

    from .config import AuralConfig, resolve_model
    from .template import TEMPLATE_NAME, register_aural_template

    os.environ["ENABLE_AUDIO_OUTPUT"] = "0"
    os.environ["USE_AUDIO_IN_VIDEO"] = "0"
    directory = resolve_model(model)
    metadata = AuralConfig.load(directory)
    values = {
        field.name: getattr(metadata, field.name)
        for field in fields(InferenceConfig)
        if hasattr(metadata, field.name)
    }
    values.update(overrides)
    config = InferenceConfig(**values)
    if config.max_n_latent_forward < 1 or config.max_new_tokens < 1:
        raise ValueError("Generation budgets must be positive")
    register_aural_template(metadata.compression_factor)
    backbone, processor = get_model_processor(
        str(directory),
        model_type="qwen2_5_omni",
        torch_dtype=torch.bfloat16,
        attn_impl=attn_impl,
        device_map=device,
        use_hf=True,
        load_model=True,
    )
    backbone.eval()
    template = get_template(
        processor,
        template_type=TEMPLATE_NAME,
        default_system=metadata.system_prompt,
        response_prefix=metadata.response_prefix,
        enable_thinking=True,
    )
    template.set_mode("transformers")
    policy = ChunkLatentPolicy(
        feature_size=config.hidden_size,
        intermediate_size=config.lp_intermediate,
        num_components=config.gmm_components,
        chunk_size=config.chunk_size,
        rank=config.lowrank,
        mixer_mult=config.mixer_mult,
        min_log_std=config.gmm_min_log_std,
        max_log_std=config.gmm_max_log_std,
    )
    policy.load_state_dict(load_file(str(directory / "latent_policy.safetensors")), strict=True)
    thinker = getattr(backbone, "thinker", backbone)
    policy.to(device=_text_input_device(thinker.model), dtype=torch.bfloat16)
    policy.eval().requires_grad_(False)
    return AuralGenerator(backbone, template, policy, config)


@dataclass
class InferenceConfig:
    embeds_std: float
    hidden_size: int
    lp_intermediate: int
    gmm_components: int
    gmm_min_log_std: float
    gmm_max_log_std: float
    chunk_size: int
    lowrank: int
    mixer_mult: int
    separator_id: int
    compression_factor: int = 4
    max_n_latent_forward: int = 100
    latent_temperature: float = 1.0
    latent_decode: str = "auto"
    eol_temperature: float = 1.0
    max_new_tokens: int = 512
    answer_temperature: float = 0.0
    answer_top_p: float = 1.0
    progress_every: int = 0


LATENT_DECODE_CHOICES = ("auto", "sample", "mean", "mode")


def resolve_latent_decode(latent_decode: str, latent_temperature: float) -> str:
    if latent_decode not in LATENT_DECODE_CHOICES:
        raise ValueError(
            f"latent_decode must be one of {LATENT_DECODE_CHOICES}, got {latent_decode!r}."
        )
    if latent_decode != "auto":
        return latent_decode
    return "sample" if latent_temperature > 0 else "mean"


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    prompt_tokens: int
    n_latent_forward: int
    hit_eol: bool
    n_answer_tokens: int
    hit_answer_eos: bool
    n_chunk_forward: int = 0


def _module_device(module) -> torch.device:
    for parameter in module.parameters(recurse=True):
        return parameter.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device("cpu")


def _text_input_device(text_model) -> torch.device:
    layers = getattr(text_model, "layers", None)
    if layers is not None and len(layers):
        return _module_device(layers[0])
    return _module_device(text_model)


def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    return torch.clamp_min(attention_mask.long().cumsum(dim=1) - 1, 0)


def _sample_token(logits: torch.Tensor, temperature: float, top_p: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        if not 0 < top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}.")
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative - sorted_probs >= top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled_sorted = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(-1, sampled_sorted)
    return torch.multinomial(probabilities, num_samples=1)


def _truncate_cache(past_key_values, length: int) -> None:
    crop = getattr(past_key_values, "crop", None)
    if crop is None:
        raise RuntimeError(
            f"{type(past_key_values).__name__} has no crop(); the chunked rollout needs to discard the unused tail of a chunk after EOL. Use a cache implementation that supports cropping, or run with chunk_size=1."
        )
    crop(length)


def _decode_latent_chunk(
    mixture: ChunkMixtureOutput, temperature: float, decode: str
) -> torch.Tensor:
    if decode == "mean":
        return mixture.mean
    if decode == "mode":
        return mixture.mode
    if temperature <= 0:
        raise ValueError(
            "latent_decode='sample' requires latent_temperature > 0; use latent_decode='mean' for a deterministic rollout."
        )
    return mixture.rsample(temperature=temperature)


class AuralGenerator:
    def __init__(self, model, template, latent_policy: ChunkLatentPolicy, config: InferenceConfig):
        self.model = model
        self.template = template
        self.latent_policy = latent_policy
        self.config = config
        self.latent_decode = resolve_latent_decode(config.latent_decode, config.latent_temperature)
        self.tokenizer = template.tokenizer
        base_model = template.get_base_model(model)
        self.thinker = getattr(base_model, "thinker", base_model)
        self.text_model = self.thinker.model
        self.lm_head = self.thinker.lm_head
        self.embed_tokens = self.thinker.get_input_embeddings()
        self.embed_device = self.embed_tokens.weight.device
        self.text_device = _text_input_device(self.text_model)
        self.lm_head_device = self.lm_head.weight.device
        self.lp_device = next(self.latent_policy.parameters()).device
        self.lp_dtype = next(self.latent_policy.parameters()).dtype
        self.audio_token_id = resolve_audio_token_id(self.thinker, self.tokenizer, base_model)
        actual_separator = self.tokenizer.encode("###", add_special_tokens=False)
        if actual_separator != [config.separator_id]:
            raise ValueError(
                f"Runtime tokenizer encodes ### as {actual_separator}, checkpoint expects {[config.separator_id]}."
            )

    def _encode_prompt(self, messages, audios):
        self.template.set_mode("transformers")
        response_prefix = f"(Thinking speed: {self.config.compression_factor})###"
        encoded = self.template.encode(
            InferRequest(
                messages=messages,
                audios=audios or [],
                chat_template_kwargs={"enable_thinking": True, "response_prefix": response_prefix},
            )
        )
        input_ids = torch.tensor([encoded["input_ids"]], device=self.embed_device, dtype=torch.long)
        if int(input_ids[0, -1].item()) != self.config.separator_id:
            suffix = input_ids[0, -16:].tolist()
            raise ValueError(
                f"Qwen2.5 AURAL prompt must end at the opening ### anchor; got suffix token ids {suffix}."
            )
        if int((input_ids == self.config.separator_id).sum().item()) != 1:
            raise ValueError("Inference prompt must contain exactly one opening ### separator.")
        attention_mask = torch.ones_like(input_ids)
        input_features = encoded.get("input_features")
        feature_attention_mask = encoded.get("feature_attention_mask")
        return (input_ids, attention_mask, input_features, feature_attention_mask)

    def _prompt_embeds(self, input_ids, input_features, feature_attention_mask):
        inputs_embeds = self.embed_tokens(input_ids)
        return merge_audio_embeds(
            thinker=self.thinker,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_token_id=self.audio_token_id,
        )

    def _append_embeddings(self, embeddings, attention_mask, position_ids, past_key_values):
        n = embeddings.shape[1]
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((embeddings.shape[0], n), device=self.text_device, dtype=torch.long),
            ],
            dim=1,
        )
        position_ids = position_ids[:, -1:] + torch.arange(
            1, n + 1, device=position_ids.device
        ).reshape(1, n)
        output = self.text_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return (output, attention_mask, position_ids)

    @torch.inference_mode()
    def generate(
        self,
        messages: List[Dict[str, Any]],
        audios: Optional[List[str]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> GenerationResult:
        cfg = self.config
        answer_limit = cfg.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        if answer_limit < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {answer_limit}.")
        input_ids, attention_mask, input_features, feature_attention_mask = self._encode_prompt(
            messages, audios
        )
        prompt_tokens = int(attention_mask.sum().item())
        question_embeds = self._prompt_embeds(input_ids, input_features, feature_attention_mask).to(
            self.text_device
        )
        attention_mask = attention_mask.to(self.text_device)
        position_ids = _position_ids(attention_mask)
        output = self.text_model(
            inputs_embeds=question_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        hidden = output.last_hidden_state
        current_position_ids = position_ids[:, -1:]
        n_latent_forward = 0
        n_chunk_forward = 0
        hit_eol = False
        while n_latent_forward < cfg.max_n_latent_forward:
            latent_hidden = hidden[:, -1:, :].to(device=self.lp_device, dtype=self.lp_dtype)
            mixture = self.latent_policy(latent_hidden)
            decoded = _decode_latent_chunk(mixture, cfg.latent_temperature, self.latent_decode)
            chunk = decoded.reshape(decoded.shape[0], cfg.chunk_size, -1)
            budget = cfg.max_n_latent_forward - n_latent_forward
            if chunk.shape[1] > budget:
                chunk = chunk[:, :budget]
            latent_embeddings = chunk.to(self.text_device, question_embeds.dtype) * cfg.embeds_std
            output, attention_mask, current_position_ids = self._append_embeddings(
                latent_embeddings, attention_mask, current_position_ids, past_key_values
            )
            past_key_values = output.past_key_values
            hidden = output.last_hidden_state
            n_chunk_forward += 1
            eol_logits = self.lm_head(hidden.to(self.lm_head_device))
            accepted = hidden.shape[1]
            for slot in range(hidden.shape[1]):
                token = _sample_token(eol_logits[:, slot, :], cfg.eol_temperature)
                if int(token.item()) == cfg.separator_id:
                    hit_eol = True
                    accepted = slot + 1
                    break
            n_latent_forward += accepted
            if hit_eol:
                surplus = hidden.shape[1] - accepted
                if surplus:
                    _truncate_cache(past_key_values, attention_mask.shape[1] - surplus)
                    attention_mask = attention_mask[:, : attention_mask.shape[1] - surplus]
                    current_position_ids = current_position_ids[:, :accepted]
                hidden = hidden[:, :accepted]
                break
        close_id = torch.tensor([[cfg.separator_id]], device=self.embed_device, dtype=torch.long)
        close_embedding = self.embed_tokens(close_id).to(self.text_device)
        output, attention_mask, current_position_ids = self._append_embeddings(
            close_embedding, attention_mask, current_position_ids, past_key_values
        )
        past_key_values = output.past_key_values
        hidden = output.last_hidden_state
        generated_ids: List[int] = []
        eos_token_id = self.tokenizer.eos_token_id
        hit_answer_eos = False
        for _ in range(answer_limit):
            answer_logits = self.lm_head(hidden[:, -1:, :].to(self.lm_head_device))[:, -1, :]
            next_token = _sample_token(answer_logits, cfg.answer_temperature, cfg.answer_top_p)
            token_id = int(next_token.item())
            if eos_token_id is not None and token_id == eos_token_id:
                hit_answer_eos = True
                break
            generated_ids.append(token_id)
            if cfg.progress_every and len(generated_ids) % cfg.progress_every == 0:
                print(f"[qwen25-chunk] answer decode: {len(generated_ids)}/{answer_limit}")
            next_embedding = self.embed_tokens(
                next_token.to(self.embed_device, dtype=torch.long)
            ).to(self.text_device)
            output, attention_mask, current_position_ids = self._append_embeddings(
                next_embedding, attention_mask, current_position_ids, past_key_values
            )
            past_key_values = output.past_key_values
            hidden = output.last_hidden_state
        answer = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return GenerationResult(
            answer=clean_answer(answer),
            prompt_tokens=prompt_tokens,
            n_latent_forward=n_latent_forward,
            hit_eol=hit_eol,
            n_answer_tokens=len(generated_ids),
            hit_answer_eos=hit_answer_eos,
            n_chunk_forward=n_chunk_forward,
        )

    def latent_generate(
        self, messages: List[Dict[str, Any]], audios: Optional[List[str]] = None
    ) -> Tuple[str, int, bool]:
        result = self.generate(messages, audios=audios)
        return (result.answer, result.n_latent_forward, result.hit_eol)
