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

    @staticmethod
    def _cfg_value(cfg: Any, key: str) -> Any:
        value = getattr(cfg, key, None)
        if value in (None, ""):
            return None
        return value

    def _resolve_run_id(self, state: dict) -> Optional[str]:
        configured = self._cfg_value(self.logger_cfg, "run_id")
        if configured is not None:
            return str(configured)
        saved = state.get("wandb_run_id", None)
        return str(saved) if saved not in (None, "") else None

    def _resolve_resume_mode(self, state: dict, run_id: Optional[str]) -> Optional[str]:
        configured = self._cfg_value(self.logger_cfg, "resume")
        if configured is not None:
            return str(configured)
        # Checkpoint resume should automatically attempt to reattach to the
        # stored W&B run when a prior run id is available.
        if run_id is not None and state.get("wandb_run_id", None) not in (None, ""):
            return "allow"
        return None

    def _resolve_run_name(self, state: dict, run_id: Optional[str]) -> str:
        configured = str(self.logger_cfg.run_name)
        if run_id is None:
            return configured
        if not bool(getattr(self.logger_cfg, "resume_use_saved_name", True)):
            return configured
        saved = state.get("wandb_run_name", None)
        if saved not in (None, ""):
            return str(saved)
        return configured

    def on_run_start(self, cfg: Any, state: dict, model: Any) -> None:
        if not is_rank0():
            return
        if wandb is None:
            raise RuntimeError(
                "WandbCallback selected (cfg.logger.name='wandb') but 'wandb' is not installed. "
                "Install it with: pip install wandb"
            )

        full_cfg = OmegaConf.to_container(cfg, resolve=False)
        run_id = self._resolve_run_id(state)
        resume_mode = self._resolve_resume_mode(state, run_id)
        run_name = self._resolve_run_name(state, run_id)
        init_kwargs = dict(
            project=str(self.logger_cfg.project),
            entity=getattr(self.logger_cfg, "entity", None),
            name=run_name,
            group=str(getattr(self.logger_cfg, "group", "")) or None,
            tags=list(getattr(self.logger_cfg, "tags", [])) or None,
            notes=getattr(self.logger_cfg, "notes", None),
            mode=str(getattr(self.logger_cfg, "mode", "online")),
            config=full_cfg,
        )
        if run_id is not None:
            init_kwargs["id"] = run_id
        if resume_mode is not None:
            init_kwargs["resume"] = resume_mode
        self.run = wandb.init(**init_kwargs)
        state["wandb_run_id"] = str(self.run.id)
        state["wandb_run_name"] = str(self.run.name)

    @staticmethod
    def _convert_histograms(metrics: dict) -> dict:
        """Convert ``_hist/`` keys (numpy arrays) to ``wandb.Histogram``."""
        out = {}
        for k, v in metrics.items():
            if str(k).startswith("_hist/"):
                clean_key = k[len("_hist/"):]
                out[clean_key] = wandb.Histogram(v)
            else:
                out[k] = v
        return out

    def on_step_end(self, cfg: Any, state: dict, metrics: Dict[str, float]) -> None:
        if self.run is None or not is_rank0():
            return
        wandb.log(self._convert_histograms(metrics), step=int(state["global_step"]))

    def on_before_train_start(
        self, cfg: Any, state: dict, metrics: Dict[str, float]
    ) -> None:
        if self.run is None or not is_rank0() or not metrics:
            return
        wandb.log(self._convert_histograms(metrics), step=int(state["global_step"]))

    def on_epoch_end(self, cfg: Any, state: dict, metrics: Dict[str, float]) -> None:
        if self.run is None or not is_rank0():
            return
        wandb.log(self._convert_histograms(metrics), step=int(state["global_step"]))

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
