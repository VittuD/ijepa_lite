from __future__ import annotations

import inspect
from typing import Any, Type

from ijepa_lite.masking.base import LatentMasker

_REGISTRY: dict[str, Type[LatentMasker]] = {}


def register(name: str):
    """
    Class decorator that registers a LatentMasker subclass by name.

    Usage
    -----
    @register("my_masker")
    class MyMasker(LatentMasker):
        ...

    The registered name is then referenced in the config:
        masking:
          latent:
            name: my_masker
    """

    def decorator(cls: Type[LatentMasker]) -> Type[LatentMasker]:
        if not issubclass(cls, LatentMasker):
            raise TypeError(
                f"@register can only decorate LatentMasker subclasses, got {cls!r}"
            )
        if name in _REGISTRY:
            raise ValueError(
                f"LatentMasker name={name!r} is already registered "
                f"(existing: {_REGISTRY[name]!r}, new: {cls!r}). Use a unique name."
            )
        _REGISTRY[name] = cls
        return cls

    return decorator


def build_latent_masker(name: str, **kwargs: Any) -> LatentMasker:
    """
    Instantiate a registered LatentMasker by name.

    Kwargs filtering
    ----------------
    build.py's auto_kwargs is intentionally a superset of what any single masker
    needs — it includes predictor-compatible fields, loss-related fields, and
    model-level fields.  Rather than requiring every masker to accept **kwargs,
    this function inspects the constructor signature and silently drops any kwarg
    that is not explicitly accepted.

    This means:
      - GumbelTopKMasker sees:     dim, num_patches, target_ratio, context_ratio,
                                   temperature, entropy_coeff
      - PredictorBasedMasker sees: + predictor_dim, depth, num_heads, mlp_ratio, dropout
      - RateDist3WayMasker sees:   + base_kind, normalize, lam_min, lam_max
      - Any future masker:         only what its __init__ declares

    Raises
    ------
    ValueError  if name is not registered.
    TypeError   if a *required* constructor argument (no default) is missing after
                filtering — this is intentional and surfaces real config errors.
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ValueError(
            f"LatentMasker name={name!r} is not registered. "
            f"Available: {available}. "
            f"Decorate your class with @register({name!r}) and import its module "
            f"before build_latent_masker is called."
        )

    cls = _REGISTRY[name]
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}

    # Check if constructor accepts **kwargs — if so, pass everything through
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )

    filtered = kwargs if has_var_keyword else {
        k: v for k, v in kwargs.items() if k in valid_params
    }

    return cls(**filtered)


def registered_names() -> list[str]:
    """Return all currently registered masker names (sorted)."""
    return sorted(_REGISTRY.keys())