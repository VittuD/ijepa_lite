from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ijepa_lite.utils.dist import all_reduce_sum


class EppsPulley1D(nn.Module):
    """
    Approximate an Epps-Pulley-style normality statistic on projected samples.

    The input is expected to have shape (N, S), where N is the number of
    samples and S is the number of projection slices.  The same integration
    grid is reused for all slices, and distributed workers aggregate their
    sample moments so the statistic matches the global batch.
    """

    def __init__(
        self,
        num_t: int = 16,
        t_max: float = 4.0,
    ) -> None:
        super().__init__()
        if num_t <= 0:
            raise ValueError("num_t must be > 0")
        if t_max <= 0.0:
            raise ValueError("t_max must be > 0")
        self.num_t = int(num_t)
        self.t_max = float(t_max)
        self.register_buffer("_t", torch.empty(0), persistent=False)
        self.register_buffer("_weights", torch.empty(0), persistent=False)

    def _get_grid(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._t.numel() == 0
            or self._weights.numel() == 0
            or self._t.device != device
        ):
            t = torch.linspace(
                0.0,
                self.t_max,
                steps=self.num_t + 1,
                device=device,
                dtype=torch.float32,
            )[1:]
            dt = self.t_max / float(self.num_t)
            weights = torch.full_like(t, dt)
            self._t = t
            self._weights = weights
        return self._t, self._weights

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(f"Expected x to have shape (N, S), got {tuple(x.shape)}")

        x = x.float()
        t, weights = self._get_grid(x.device)

        xt = x.unsqueeze(-1) * t.view(1, 1, -1)  # (N, S, T)
        cos_local = torch.cos(xt).sum(dim=0)  # (S, T)
        sin_local = torch.sin(xt).sum(dim=0)  # (S, T)

        sample_count = torch.tensor(float(x.shape[0]), device=x.device)
        cos_sum = all_reduce_sum(cos_local)
        sin_sum = all_reduce_sum(sin_local)
        n_total = all_reduce_sum(sample_count).clamp(min=1.0)

        cos_mean = cos_sum / n_total
        sin_mean = sin_sum / n_total
        phi_normal = torch.exp(-0.5 * t.square()).view(1, -1)  # (1, T)

        err = (cos_mean - phi_normal).square() + sin_mean.square()
        stat_per_slice = n_total * (err * weights.view(1, -1)).sum(dim=-1)  # (S,)
        return stat_per_slice, n_total


class SIGRegLoss(nn.Module):
    """
    Multivariate Gaussianity regularizer via random 1D projections.

    A deterministic projection bank is used so every distributed worker
    evaluates the same slices before global aggregation.
    """

    def __init__(
        self,
        num_slices: int = 16,
        num_t: int = 16,
        t_max: float = 4.0,
        standardize: bool = True,
        eps: float = 1e-6,
        projection_seed: int = 0,
    ) -> None:
        super().__init__()
        if num_slices <= 0:
            raise ValueError("num_slices must be > 0")
        self.num_slices = int(num_slices)
        self.standardize = bool(standardize)
        self.eps = float(eps)
        self.projection_seed = int(projection_seed)
        self.ep_test = EppsPulley1D(num_t=num_t, t_max=t_max)
        self.register_buffer("_projection", torch.empty(0), persistent=False)

    def _get_projection(self, dim: int, device: torch.device) -> torch.Tensor:
        if (
            self._projection.numel() == 0
            or self._projection.shape != (self.num_slices, dim)
            or self._projection.device != device
        ):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.projection_seed)
            proj = torch.randn(
                self.num_slices,
                dim,
                generator=gen,
                dtype=torch.float32,
            )
            proj = F.normalize(proj, dim=-1, eps=self.eps)
            self._projection = proj.to(device=device)
        return self._projection

    def _standardize_tokens(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local_count = torch.tensor(float(z.shape[0]), device=z.device)
        sum_local = z.sum(dim=0)
        sq_sum_local = z.square().sum(dim=0)

        count = all_reduce_sum(local_count).clamp(min=1.0)
        sum_total = all_reduce_sum(sum_local)
        sq_sum_total = all_reduce_sum(sq_sum_local)

        mean = sum_total / count
        var = (sq_sum_total / count) - mean.square()
        var = var.clamp(min=self.eps)
        z_std = (z - mean) / var.sqrt()
        return z_std, var.sqrt().mean()

    def forward(self, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if patch_tokens.ndim != 3:
            raise ValueError(
                "SIGRegLoss expects patch tokens with shape (B, N, D); "
                f"got {tuple(patch_tokens.shape)}"
            )

        z = patch_tokens.float().reshape(-1, patch_tokens.shape[-1])  # (B*N, D)
        pre_std_mean = z.std(dim=0, correction=0).mean()

        post_std_mean = pre_std_mean
        if self.standardize:
            z, post_std_mean = self._standardize_tokens(z)

        proj = self._get_projection(z.shape[-1], z.device)  # (S, D)
        projected = z @ proj.transpose(0, 1)  # (B*N, S)

        stat_per_slice, n_total = self.ep_test(projected)
        loss = stat_per_slice.mean()

        logs = {
            "sigreg/loss": float(loss.detach().item()),
            "sigreg/stat_mean": float(stat_per_slice.detach().mean().item()),
            "sigreg/stat_max": float(stat_per_slice.detach().amax().item()),
            "sigreg/num_samples": float(n_total.detach().item()),
            "sigreg/pre_token_std_mean": float(pre_std_mean.detach().item()),
            "sigreg/post_token_std_mean": float(post_std_mean.detach().item()),
        }
        return loss, logs
