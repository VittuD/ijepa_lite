"""
Composite masker loss — weighted sum of atomic terms.

No learned parameters. No sampling logic. Pure weighted sum.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ijepa_lite.losses.terms import TERM_REGISTRY


class CompositeMaskerLoss(nn.Module):
    """
    Instantiates active terms from a config dict and computes their weighted sum.

    Args
    ----
    terms_cfg  : Nested dict from config, e.g.::

            {"H_cond": {"weight": [0.01, 1.0]},
             "floor_penalty": {"weight": [5.0, 5.0], "h_floor": 0.1},
             "ignore_tax": {"weight": [0, 0]}}  # disabled

    num_patches : N — total patch positions (forwarded to terms that need it).
    """

    def __init__(self, terms_cfg: dict, num_patches: int) -> None:
        super().__init__()

        self.weight_ranges: dict[str, tuple[float, float]] = {}
        self.terms = nn.ModuleDict()

        for name, tcfg in terms_cfg.items():
            w = tcfg["weight"]
            lo, hi = float(w[0]), float(w[1])
            if lo == 0.0 and hi == 0.0:
                continue  # disabled

            self.weight_ranges[name] = (lo, hi)

            # Collect extra kwargs for the term (everything except "weight")
            extra = {k: v for k, v in tcfg.items() if k != "weight"}

            cls = TERM_REGISTRY[name]

            # Inject num_patches if the term accepts it
            import inspect
            sig = inspect.signature(cls.__init__)
            if "num_patches" in sig.parameters:
                extra["num_patches"] = num_patches

            self.terms[name] = cls(**extra)

    def forward(
        self,
        weights: dict[str, float],
        p_ctx: torch.Tensor,
        p_tgt: torch.Tensor,
        p_ign: torch.Tensor,
        ema_full: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns
        -------
        total : scalar loss (weighted sum of active terms)
        logs  : dict of diagnostic values from each term + derived mi_rate
        """
        total = torch.tensor(0.0, device=p_ctx.device)
        logs: dict[str, float] = {}

        for name, term in self.terms.items():
            w = weights.get(name, 0.0)
            if w == 0.0:
                continue
            val, term_logs = term(
                p_ctx=p_ctx, p_tgt=p_tgt, p_ign=p_ign, ema_full=ema_full,
            )
            total = total + w * val
            logs.update(term_logs)

        # Derived metric: mi_rate = H_cond - H_marg (for backwards compat)
        if "entropy_conditional" in logs and "entropy_marginal" in logs:
            logs["mi_rate"] = logs["entropy_conditional"] - logs["entropy_marginal"]

        return total, logs
