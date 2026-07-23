from __future__ import annotations

import pytest
import torch

from ijepa_lite.masking.base import LatentMasker, MaskOutput
from ijepa_lite.masking.registry import build_latent_masker, register


@register("test_strict_config")
class _StrictConfigMasker(LatentMasker):
    def __init__(self, dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.temperature = float(temperature)

    def forward(
        self,
        tokens: torch.Tensor,
        ema_full: torch.Tensor | None = None,
    ) -> MaskOutput:
        raise NotImplementedError


def test_inferred_constructor_superset_is_filtered() -> None:
    masker = build_latent_masker(
        "test_strict_config",
        inferred_kwargs={"dim": 8, "unused_model_default": 123},
    )

    assert isinstance(masker, _StrictConfigMasker)
    assert masker.dim == 8


def test_explicit_config_overrides_inferred_value() -> None:
    masker = build_latent_masker(
        "test_strict_config",
        inferred_kwargs={"dim": 8, "temperature": 1.0},
        temperature=0.25,
    )

    assert masker.temperature == 0.25


def test_unsupported_explicit_config_field_fails_loudly() -> None:
    with pytest.raises(TypeError, match=r"test_strict_config.*ema_signal"):
        build_latent_masker(
            "test_strict_config",
            inferred_kwargs={"dim": 8},
            ema_signal=True,
        )


def test_unimplemented_goldilocks_ema_signal_cannot_run_silently() -> None:
    with pytest.raises(TypeError, match=r"goldilocks.*ema_signal"):
        build_latent_masker("goldilocks", ema_signal=True)
