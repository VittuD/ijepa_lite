from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ijepa_lite.callbacks.base import Callback
from ijepa_lite.utils.dist import is_rank0


def _fmt(v: object) -> str:
    if isinstance(v, bool):
        return str(v)

    # ints stay ints
    if isinstance(v, int):
        return str(v)

    if isinstance(v, float):
        # treat exact integer-valued floats like ints (e.g., 0.0, 1.0)
        if v.is_integer():
            return str(int(v))
        return f"{v:.3e}"

    return str(v)


@dataclass
class ProgressCallback(Callback):
    log_every: int = 50

    def on_step_end(self, cfg: Any, state: dict, metrics: Dict[str, float]) -> None:
        if not is_rank0():
            return
        step = int(state["global_step"])
        if step % int(self.log_every) == 0:
            msg = " ".join([f"{k}={_fmt(v)}" for k, v in metrics.items()])
            print(f"[step {step}] {msg}")

    def on_epoch_end(self, cfg: Any, state: dict, metrics: Dict[str, float]) -> None:
        if not is_rank0():
            return
        msg = " ".join([f"{k}={_fmt(v)}" for k, v in metrics.items()])
        print(f"[epoch {state['epoch']}] {msg}")
