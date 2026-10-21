import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ChunkMixtureOutput:
    def __init__(
        self,
        mixture_logits: torch.Tensor,
        means: torch.Tensor,
        log_stds: torch.Tensor,
        time_loadings: Optional[torch.Tensor],
        feature_directions: Optional[torch.Tensor],
    ) -> None:
        self.mixture_logits = mixture_logits
        self.means = means
        self.log_stds = log_stds
        self.time_loadings = time_loadings
        self.feature_directions = feature_directions

    @property
    def rank(self) -> int:
        return 0 if self.time_loadings is None else int(self.time_loadings.shape[-1])

    @property
    def chunk_size(self) -> int:
        return int(self.means.shape[-2])

    @property
    def feature_size(self) -> int:
        return int(self.means.shape[-1])

    @property
    def probabilities(self) -> torch.Tensor:
        return torch.softmax(self.mixture_logits.float(), dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        weights = self.probabilities.to(self.means.dtype)[..., None, None]
        return (weights * self.means).sum(dim=-3)

    @property
    def mode(self) -> torch.Tensor:
        return self._gather_component(self.mixture_logits.argmax(dim=-1))[0]

    def _gather_component(self, component: torch.Tensor):
        m, feature = (self.chunk_size, self.feature_size)
        index = component[..., None, None, None].expand(*component.shape, 1, m, feature)
        mean = self.means.gather(dim=-3, index=index).squeeze(-3)
        log_std = self.log_stds.gather(dim=-3, index=index).squeeze(-3)
        loadings = None
        if self.time_loadings is not None:
            rank = self.rank
            loading_index = component[..., None, None, None].expand(*component.shape, 1, m, rank)
            loadings = self.time_loadings.gather(dim=-3, index=loading_index).squeeze(-3)
        return (mean, log_std, loadings)

    def _woodbury_terms(
        self, residual: torch.Tensor, inverse_variance: torch.Tensor, loadings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        directions = self.feature_directions.float()
        rank = directions.shape[-1]
        projected = torch.einsum("...sh,hj->...sj", residual * inverse_variance, directions)
        f_u = (loadings * projected).sum(dim=-2)
        outer = directions.unsqueeze(-1) * directions.unsqueeze(-2)
        gram = torch.einsum("...sh,hjk->...sjk", inverse_variance, outer)
        f_f = torch.einsum("...sj,...sk,...sjk->...jk", loadings, loadings, gram)
        capacitance = f_f + torch.eye(rank, device=f_f.device, dtype=f_f.dtype)
        return (f_u, capacitance)

    def component_log_prob(
        self, value: torch.Tensor, slot_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        value = value.float().unsqueeze(-3)
        means = self.means.float()
        log_stds = self.log_stds.float()
        residual = value - means
        inverse_variance = torch.exp(-2.0 * log_stds)
        if slot_mask is not None:
            mask = slot_mask.float()[..., None, :, None]
            residual = residual * mask
            inverse_variance = inverse_variance * mask
            dimension = slot_mask.float().sum(dim=-1)[..., None] * self.feature_size
            log_det = (2.0 * log_stds * mask).sum(dim=(-1, -2))
        else:
            dimension = torch.full(
                self.mixture_logits.shape,
                float(self.chunk_size * self.feature_size),
                device=value.device,
                dtype=torch.float32,
            )
            log_det = (2.0 * log_stds).sum(dim=(-1, -2))
        quadratic = (residual.square() * inverse_variance).sum(dim=(-1, -2))
        if self.time_loadings is not None and self.rank > 0:
            f_u, capacitance = self._woodbury_terms(
                residual, inverse_variance, self.time_loadings.float()
            )
            cholesky = torch.linalg.cholesky(capacitance)
            log_det = log_det + 2.0 * torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)).sum(
                dim=-1
            )
            solved = torch.cholesky_solve(f_u.unsqueeze(-1), cholesky).squeeze(-1)
            quadratic = quadratic - (f_u * solved).sum(dim=-1)
        return -0.5 * (dimension * math.log(2.0 * math.pi) + log_det + quadratic)

    def log_prob(
        self, value: torch.Tensor, slot_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        log_weights = torch.log_softmax(self.mixture_logits.float(), dim=-1)
        return torch.logsumexp(log_weights + self.component_log_prob(value, slot_mask), dim=-1)

    def entropy_upper_bound(self) -> torch.Tensor:
        probabilities = self.probabilities
        categorical = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        log_det = (2.0 * self.log_stds.float()).sum(dim=(-1, -2))
        if self.time_loadings is not None and self.rank > 0:
            inverse_variance = torch.exp(-2.0 * self.log_stds.float())
            _, capacitance = self._woodbury_terms(
                torch.zeros_like(inverse_variance), inverse_variance, self.time_loadings.float()
            )
            log_det = log_det + torch.logdet(capacitance)
        dimension = float(self.chunk_size * self.feature_size)
        component = 0.5 * (log_det + dimension * math.log(2.0 * math.pi * math.e))
        return categorical + (probabilities * component).sum(dim=-1)

    def rsample(self, temperature: float = 1.0) -> torch.Tensor:
        if temperature <= 0:
            return self.mean
        logits = self.mixture_logits.float() / float(temperature)
        component = torch.distributions.Categorical(logits=logits).sample()
        mean, log_std, loadings = self._gather_component(component)
        std = torch.exp(log_std.float()).to(mean.dtype)
        sample = mean + torch.randn_like(mean) * std * float(temperature)
        if loadings is not None and self.rank > 0:
            directions = self.feature_directions.to(mean.dtype)
            factor_noise = torch.randn(
                (*loadings.shape[:-2], 1, self.rank), device=mean.device, dtype=mean.dtype
            )
            correlated = torch.einsum(
                "...sj,hj->...sh", loadings.to(mean.dtype) * factor_noise, directions
            )
            sample = sample + correlated * float(temperature)
        return sample

    def sample(self, temperature: float = 1.0) -> torch.Tensor:
        if temperature <= 0:
            return self.mode
        return self.rsample(temperature=temperature)


class ChunkLatentPolicy(nn.Module):
    def __init__(
        self,
        feature_size: int,
        intermediate_size: int = 512,
        num_components: int = 4,
        chunk_size: int = 4,
        rank: int = 16,
        mixer_mult: int = 1,
        min_log_std: float = -4.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        if num_components < 2:
            raise ValueError(f"GMM requires at least two components, got {num_components}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if rank < 0:
            raise ValueError(f"rank must be non-negative, got {rank}")
        if mixer_mult < 1:
            raise ValueError(f"mixer_mult must be positive, got {mixer_mult}")
        if min_log_std >= max_log_std:
            raise ValueError(
                f"min_log_std must be below max_log_std, got {min_log_std} >= {max_log_std}"
            )
        self.feature_size = int(feature_size)
        self.num_components = int(num_components)
        self.chunk_size = int(chunk_size)
        self.rank = int(rank)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.fc = nn.Sequential(
            nn.Linear(feature_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, intermediate_size),
            nn.LayerNorm(intermediate_size),
        )
        self.slot_embedding = nn.Parameter(torch.zeros(chunk_size, intermediate_size))
        nn.init.normal_(self.slot_embedding, std=0.02)
        self.slot_norm = nn.LayerNorm(intermediate_size)
        self.slot_token_mix = nn.Linear(chunk_size, chunk_size)
        self.slot_channel_norm = nn.LayerNorm(intermediate_size)
        self.slot_channel_mix = nn.Sequential(
            nn.Linear(intermediate_size, intermediate_size * mixer_mult),
            nn.GELU(),
            nn.Linear(intermediate_size * mixer_mult, intermediate_size),
        )
        nn.init.zeros_(self.slot_token_mix.weight)
        nn.init.zeros_(self.slot_token_mix.bias)
        nn.init.zeros_(self.slot_channel_mix[-1].weight)
        nn.init.zeros_(self.slot_channel_mix[-1].bias)
        self.means = nn.Linear(intermediate_size, num_components * feature_size)
        self.log_stds = nn.Linear(intermediate_size, num_components * feature_size)
        self.mixture_logits = nn.Linear(intermediate_size, num_components)
        if rank > 0:
            self.time_loadings = nn.Linear(intermediate_size, num_components * rank)
            self.feature_directions = nn.Parameter(torch.empty(feature_size, rank))
            nn.init.normal_(self.feature_directions, std=1.0 / math.sqrt(feature_size))
            nn.init.zeros_(self.time_loadings.weight)
            nn.init.zeros_(self.time_loadings.bias)
        else:
            self.time_loadings = None
            self.feature_directions = None
        nn.init.zeros_(self.log_stds.weight)
        nn.init.zeros_(self.log_stds.bias)
        nn.init.zeros_(self.mixture_logits.weight)
        nn.init.zeros_(self.mixture_logits.bias)

    def _slots(self, pooled: torch.Tensor) -> torch.Tensor:
        slots = pooled.unsqueeze(-2) + self.slot_embedding.to(pooled.dtype)
        mixed = self.slot_norm(slots).transpose(-1, -2)
        slots = slots + self.slot_token_mix(mixed).transpose(-1, -2)
        return slots + self.slot_channel_mix(self.slot_channel_norm(slots))

    def forward(self, hidden: torch.Tensor) -> ChunkMixtureOutput:
        pooled = self.fc(hidden)
        slots = self._slots(pooled)
        leading = slots.shape[:-2]
        shape = (*leading, self.chunk_size, self.num_components, self.feature_size)
        means = self.means(slots).reshape(shape).transpose(-3, -2)
        log_stds = self.log_stds(slots).reshape(shape).transpose(-3, -2)
        log_stds = log_stds.clamp(min=self.min_log_std, max=self.max_log_std)
        mixture_logits = self.mixture_logits(pooled)
        loadings = None
        if self.rank > 0:
            loadings = (
                self.time_loadings(slots)
                .reshape(*leading, self.chunk_size, self.num_components, self.rank)
                .transpose(-3, -2)
            )
        return ChunkMixtureOutput(
            mixture_logits, means, log_stds, loadings, self.feature_directions
        )


@torch.no_grad()
def compute_embeds_std(model) -> float:
    emb = None
    try:
        emb = model.get_input_embeddings()
    except Exception:
        emb = None
    if emb is None or not hasattr(emb, "weight"):
        base = getattr(model, "thinker", model)
        inner = getattr(base, "model", base)
        emb = (
            inner.get_input_embeddings()
            if hasattr(inner, "get_input_embeddings")
            else inner.embed_tokens
        )
    w = emb.weight
    return w.float().reshape(-1).std().item()
