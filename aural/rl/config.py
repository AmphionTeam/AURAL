import os
from dataclasses import dataclass


@dataclass
class RLConfig:
    group_size: int = 8
    max_latent_forward: int = 64
    max_answer_tokens: int = 128
    latent_temperature: float = 1.0
    eol_temperature: float = 1.0
    answer_temperature: float = 1.0
    answer_top_p: float = 0.95
    clip_epsilon: float = 0.2
    latent_loss_weight: float = 1.0
    stop_loss_weight: float = 1.0
    answer_loss_weight: float = 1.0
    length_reward_weight: float = 0.5
    missing_eol_penalty: float = 0.1
    missing_eos_penalty: float = 0.02
    mixed_low: float = 0.25
    mixed_high: float = 0.75
    advantage_epsilon: float = 1e-6
    require_mixed_group: bool = True
    logit_chunk_size: int = 64
    rollout_token_budget: int = 32768

    @classmethod
    def from_env(cls):
        defaults = cls()
        values = {}
        for key, default in vars(defaults).items():
            value = os.environ.get("AURAL_RL_" + key.upper())
            if value is not None:
                values[key] = (
                    value.lower() in {"1", "true", "yes"}
                    if isinstance(default, bool)
                    else type(default)(value)
                )
        config = cls(**values)
        for key in (
            "group_size",
            "max_latent_forward",
            "max_answer_tokens",
            "rollout_token_budget",
        ):
            if getattr(config, key) < 1:
                raise ValueError(f"{key} must be positive")
        if config.group_size < 2 or not 0 < config.clip_epsilon < 1:
            raise ValueError("Invalid GRPO group size or clipping coefficient")
        return config
