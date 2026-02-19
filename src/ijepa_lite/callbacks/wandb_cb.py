from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from ijepa_lite.callbacks.base import Callback
from ijepa_lite.utils.dist import is_rank0

try:
    import wandb
except Exception:
    wandb = None


@dataclass
class WandbCallback(Callback):
    logger_cfg: Any
    run: Optional[Any] = None

    def on_run_start(self, cfg: Any, state: dict, model: Any) -> None:
        if not is_rank0():
            return
        if wandb is None:
            raise RuntimeError(
                "WandbCallback selected (cfg.logger.name='wandb') but 'wandb' is not installed. "
                "Install it with: pip install wandb"
            )

        full_cfg = OmegaConf.to_container(cfg, resolve=False)
        self.run = wandb.init(
            project=str(self.logger_cfg.project),
            entity=getattr(self.logger_cfg, "entity", None),
            name=str(self.logger_cfg.run_name),
            group=str(getattr(self.logger_cfg, "group", "")) or None,
            tags=list(getattr(self.logger_cfg, "tags", [])) or None,
            notes=getattr(self.logger_cfg, "notes", None),
            mode=str(getattr(self.logger_cfg, "mode", "online")),
            config=full_cfg,
        )

    def on_step_end(self, cfg: Any, state: dict, metrics: Dict[str, float]) -> None:
        if self.run is None or not is_rank0():
            return
        wandb.log(metrics, step=int(state["global_step"]))

    def on_epoch_end(self, cfg: Any, state: dict, metrics: Dict[str, float]) -> None:
        if self.run is None or not is_rank0():
            return
        wandb.log(metrics, step=int(state["global_step"]))

    def on_checkpoint_saved(self, cfg: Any, state: dict, path: str) -> None:
        if self.run is None or not is_rank0():
            return
        if bool(getattr(self.logger_cfg, "log_model", True)) and os.path.exists(path):
            art = wandb.Artifact(name=f"{cfg.exp_name}-checkpoint", type="checkpoint")
            art.add_file(path)
            self.run.log_artifact(art)

    def on_run_end(self, cfg: Any, state: dict) -> None:
        if self.run is None or not is_rank0():
            return
        self.run.finish()
