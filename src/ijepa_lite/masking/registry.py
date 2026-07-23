from __future__ import annotations

import inspect
from collections.abc import Mapping
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


def build_latent_masker(
    name: str,
    *,
    inferred_kwargs: Mapping[str, Any] | None = None,
    **config_kwargs: Any,
) -> LatentMasker:
    """
    Instantiate a registered LatentMasker by name.

    Argument validation
    -------------------
    ``inferred_kwargs`` is intentionally allowed to be a superset of what a
    single masker needs. Values inferred from model, predictor, and loss config
    are filtered against the selected constructor.

    Explicit ``config_kwargs`` are different: an unsupported field is almost
    certainly a typo or an unimplemented feature and therefore raises. Explicit
    values override inferred values when both provide the same constructor field.

    This means:
      - GumbelTopKMasker sees:     dim, num_patches, target_ratio, context_ratio,
                                   temperature, entropy_coeff
      - PredictorBasedMasker sees: + predictor_dim, depth, num_heads, mlp_ratio, dropout
      - RateDist3WayMasker sees:   + base_kind, normalize, lam_min, lam_max
      - Any future masker:         only what its __init__ declares

    Raises
    ------
    ValueError  if name is not registered.
    TypeError   if an explicit field is unsupported or a required constructor
                argument is missing.
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

    inferred = dict(inferred_kwargs or {})
    if has_var_keyword:
        filtered_inferred = inferred
    else:
        filtered_inferred = {
            key: value for key, value in inferred.items() if key in valid_params
        }

        unsupported = sorted(set(config_kwargs) - valid_params)
        if unsupported:
            raise TypeError(
                f"LatentMasker name={name!r} does not accept explicit config "
                f"field(s): {', '.join(unsupported)}. "
                f"Accepted fields: {', '.join(sorted(valid_params))}."
            )

    return cls(**{**filtered_inferred, **config_kwargs})


def registered_names() -> list[str]:
    """Return all currently registered masker names (sorted)."""
    return sorted(_REGISTRY.keys())
