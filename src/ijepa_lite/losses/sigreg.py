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
        self.register_buffer("_phi", torch.empty(0), persistent=False)
        self.register_buffer("_weights", torch.empty(0), persistent=False)

    def _get_grid(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self._t.numel() == 0
            or self._phi.numel() == 0
            or self._weights.numel() == 0
            or self._t.device != device
        ):
            t = torch.linspace(
                0.0,
                self.t_max,
                steps=self.num_t + 1,
                device=device,
                dtype=torch.float32,
            )
            dt = self.t_max / float(self.num_t)
            weights = torch.full_like(t, 2.0 * dt)
            weights[0] = dt
            weights[-1] = dt
            phi = torch.exp(-0.5 * t.square())
            self._t = t
            self._phi = phi
            self._weights = weights * phi
        return self._t, self._phi, self._weights

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(f"Expected x to have shape (N, S), got {tuple(x.shape)}")

        x = x.float()
        t, phi_normal, weights = self._get_grid(x.device)

        xt = x.unsqueeze(-1) * t.view(1, 1, -1)  # (N, S, T)
        cos_local = torch.cos(xt).sum(dim=0)  # (S, T)
        sin_local = torch.sin(xt).sum(dim=0)  # (S, T)

        sample_count = torch.tensor(float(x.shape[0]), device=x.device)
        cos_sum = all_reduce_sum(cos_local)
        sin_sum = all_reduce_sum(sin_local)
        n_total = all_reduce_sum(sample_count).clamp(min=1.0)

        cos_mean = cos_sum / n_total
        sin_mean = sin_sum / n_total

        err = (cos_mean - phi_normal.view(1, -1)).square() + sin_mean.square()
        stat_per_slice = n_total * (err @ weights)  # (S,)
        return stat_per_slice, n_total


class SIGRegProjector(nn.Module):
    """
    BatchNorm MLP projector matching the LeJEPA-style projected regularization path.

    The input is flattened to (B*N, D), projected, then reshaped back to (B, N, Dp).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 512,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if patch_tokens.ndim < 2:
            raise ValueError(
                "SIGRegProjector expects tokens with shape (..., D); "
                f"got {tuple(patch_tokens.shape)}"
            )

        leading_shape = patch_tokens.shape[:-1]
        z = patch_tokens.reshape(-1, patch_tokens.shape[-1])
        z = self.net(z)
        return z.reshape(*leading_shape, -1)


class SIGRegLoss(nn.Module):
    """
    Multivariate Gaussianity regularizer via random 1D projections.

    A deterministic projection bank is used so every distributed worker
    evaluates the same slices before global aggregation.
    """

    def __init__(
        self,
        input_dim: int,
        num_slices: int = 16,
        num_t: int = 16,
        t_max: float = 4.0,
        standardize: bool = False,
        eps: float = 1e-6,
        projection_seed: int = 0,
        projector_hidden_dim: int = 2048,
        projector_output_dim: int = 512,
        use_projector: bool = True,
    ) -> None:
        super().__init__()
        if num_slices <= 0:
            raise ValueError("num_slices must be > 0")
        self.num_slices = int(num_slices)
        self.standardize = bool(standardize)
        self.eps = float(eps)
        self.projection_seed = int(projection_seed)
        self.ep_test = EppsPulley1D(num_t=num_t, t_max=t_max)
        self.projector = (
            SIGRegProjector(
                input_dim=input_dim,
                hidden_dim=int(projector_hidden_dim),
                output_dim=int(projector_output_dim),
            )
            if bool(use_projector)
            else nn.Identity()
        )
        self.register_buffer("_global_step", torch.zeros((), dtype=torch.long), persistent=False)

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

    def _sample_projection(self, dim: int, device: torch.device) -> torch.Tensor:
        seed = int(self.projection_seed + int(self._global_step.item()))
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        proj = torch.randn(
            self.num_slices,
            dim,
            generator=gen,
            dtype=torch.float32,
        )
        proj = F.normalize(proj, dim=-1, eps=self.eps)
        self._global_step.add_(1)
        return proj.to(device=device)

    def project_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return self.projector(patch_tokens.float())

    def _loss_from_projected(
        self,
        projected_tokens: torch.Tensor,
        raw_std_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        z = projected_tokens.float().reshape(-1, projected_tokens.shape[-1])  # (B*N, D_proj)
        proj_std_mean = z.std(dim=0, correction=0).mean()

        post_std_mean = proj_std_mean
        if self.standardize:
            z, post_std_mean = self._standardize_tokens(z)

        proj = self._sample_projection(z.shape[-1], z.device)  # (S, D)
        projected = z @ proj.transpose(0, 1)  # (B*N, S)

        stat_per_slice, n_total = self.ep_test(projected)
        loss = stat_per_slice.mean()

        logs = {
            "sigreg/loss": float(loss.detach().item()),
            "sigreg/stat_mean": float(stat_per_slice.detach().mean().item()),
            "sigreg/stat_max": float(stat_per_slice.detach().amax().item()),
            "sigreg/num_samples": float(n_total.detach().item()),
            "sigreg/pre_token_std_mean": float(raw_std_mean.detach().item()),
            "sigreg/post_token_std_mean": float(post_std_mean.detach().item()),
            "sigreg/proj_token_std_mean": float(proj_std_mean.detach().item()),
        }
        return loss, logs

    def forward(self, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if patch_tokens.ndim != 3:
            raise ValueError(
                "SIGRegLoss expects patch tokens with shape (B, N, D); "
                f"got {tuple(patch_tokens.shape)}"
            )

        raw_tokens = patch_tokens.float()
        raw_std_mean = raw_tokens.reshape(-1, raw_tokens.shape[-1]).std(
            dim=0, correction=0
        ).mean()
        proj_tokens = self.project_tokens(raw_tokens)
        return self._loss_from_projected(proj_tokens, raw_std_mean)

    def forward_projected(
        self,
        projected_tokens: torch.Tensor,
        raw_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        raw_std_mean = raw_tokens.float().reshape(-1, raw_tokens.shape[-1]).std(
            dim=0, correction=0
        ).mean()
        return self._loss_from_projected(projected_tokens, raw_std_mean)
