from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class LatentRolloutGroup:
    prompt_embeds: torch.Tensor
    latent_values: List[torch.Tensor]
    chunk_lengths: List[List[int]]
    answer_ids: List[torch.Tensor]
    answers: List[str]
    old_latent_logps: List[torch.Tensor]
    old_stop_logps: List[torch.Tensor]
    old_answer_logps: List[torch.Tensor]
    hit_eol: torch.Tensor
    hit_eos: torch.Tensor

    @property
    def group_size(self) -> int:
        return len(self.answers)

    @property
    def latent_lengths(self) -> torch.Tensor:
        device = self.hit_eol.device
        return torch.tensor([value.shape[0] for value in self.latent_values], device=device)


@dataclass
class ActionLogProbs:
    latent: List[torch.Tensor]
    stop: List[torch.Tensor]
    answer: List[torch.Tensor]
