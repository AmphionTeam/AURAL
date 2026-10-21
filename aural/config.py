import json
from dataclasses import asdict, dataclass
from pathlib import Path

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant reasons internally before providing a helpful and accurate answer."
)
COT_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant "
    "solves it. The assistant first thinks about the reasoning process in the mind and then "
    "provides the user with the answer. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>."
)


@dataclass
class AuralConfig:
    compression_factor: int = 4
    chunk_size: int = 10
    gmm_components: int = 4
    lowrank: int = 16
    mixer_mult: int = 1
    hidden_size: int = 3584
    lp_intermediate: int = 3584
    embeds_std: float = 0.0
    gmm_min_log_std: float = -4.0
    gmm_max_log_std: float = 2.0
    separator_id: int = 14374
    lss_warmup_steps: int = 500
    system_prompt: str = SYSTEM_PROMPT
    stage: str = "sft"

    @property
    def response_prefix(self):
        return f"(Thinking speed: {self.compression_factor})###"

    def validate(self):
        for key in ("compression_factor", "chunk_size", "hidden_size", "lp_intermediate"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.gmm_components < 2 or self.lowrank < 0 or self.lss_warmup_steps < 0:
            raise ValueError("Invalid GMM or LSS configuration")
        if self.gmm_min_log_std >= self.gmm_max_log_std:
            raise ValueError("Invalid standard deviation bounds")

    def save(self, directory):
        self.validate()
        Path(directory, "aural_config.json").write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory):
        values = json.loads(Path(directory, "aural_config.json").read_text(encoding="utf-8"))
        config = cls(**values)
        config.validate()
        return config


def resolve_model(model):
    path = Path(model).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(model)))
