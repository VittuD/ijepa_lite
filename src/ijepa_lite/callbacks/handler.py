from __future__ import annotations

from typing import Any, Dict, List

from ijepa_lite.callbacks.base import Callback


class CallbackHandler:
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = list(callbacks)

    def on_run_start(self, cfg: Any, state: dict, model: Any) -> None:
        for cb in self.callbacks:
            cb.on_run_start(cfg, state, model)

    def on_epoch_start(self, cfg: Any, state: dict) -> None:
        for cb in self.callbacks:
            cb.on_epoch_start(cfg, state)

    def on_step_end(self, cfg: Any, state: dict, metrics: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_step_end(cfg, state, metrics)

    def on_epoch_end(self, cfg: Any, state: dict, metrics: Dict[str, float]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(cfg, state, metrics)

    def on_checkpoint_saved(self, cfg: Any, state: dict, path: str) -> None:
        for cb in self.callbacks:
            cb.on_checkpoint_saved(cfg, state, path)

    def on_run_end(self, cfg: Any, state: dict) -> None:
        for cb in self.callbacks:
            cb.on_run_end(cfg, state)
