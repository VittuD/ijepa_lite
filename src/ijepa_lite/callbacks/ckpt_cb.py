from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ijepa_lite.callbacks.base import Callback
from ijepa_lite.engine.checkpoint import save_checkpoint
from ijepa_lite.utils.dist import is_rank0


@dataclass
class CheckpointCallback(Callback):
    """
    Saves training checkpoints on fixed epoch and/or step cadences.

    Uses cfg.train.save_every (preferred) with fallback to save_every_epochs.
    Saves to cfg.train.ckpt_dir / cfg.train.ckpt_name with sensible defaults.

    IMPORTANT:
      - The training loop must stash runtime objects in state["_ckpt_bundle"] and
        the EMA schedule start in state["_ema_start"] (private keys).
      - This callback filters private keys out of the state that gets serialized.
      - After saving, it sets state["_checkpoint_path"] so the training loop can
        fire callbacks.on_checkpoint_saved(path) for other callbacks (e.g., W&B).

    Resume off-by-one fix:
      - state["epoch"] is the epoch that just *finished*.
      - We persist state["next_epoch"] = epoch + 1 so that on resume,
        train_loop.py reads next_epoch directly rather than re-running epoch.
    """

    save_every: int = 1
    ckpt_dir: str = "checkpoints"
    ckpt_name: str = "last.pt"
    _last_save_step: int = -1

    def on_epoch_end(self, cfg: Any, state: dict, metrics: Dict[str, float]) -> None:
        # We only checkpoint pretraining runs.
        if str(getattr(cfg.task, "name", "")).lower() != "pretrain":
            return
        if not is_rank0():
            return

        save_every = int(
            getattr(
                cfg.train,
                "save_every",
                getattr(cfg.train, "save_every_epochs", self.save_every),
            )
        )
        if save_every <= 0:
            return

        epoch = int(state.get("epoch", 0))
        if ((epoch + 1) % save_every) != 0:
            return

        self._save(cfg, state, next_epoch=epoch + 1, version_label=f"epoch_{epoch:05d}")

    def on_step_end(self, cfg: Any, state: dict, metrics: Dict[str, float]) -> None:
        # Step checkpoints are opt-in. Epoch checkpoints remain the default.
        if str(getattr(cfg.task, "name", "")).lower() != "pretrain":
            return
        if not is_rank0():
            return

        save_every_steps = int(getattr(cfg.train, "save_every_steps", 0))
        if save_every_steps <= 0:
            return

        step = int(state.get("global_step", 0))
        if step <= 0 or step == self._last_save_step:
            return
        if (step % save_every_steps) != 0:
            return
        if bool(state.get("_prefer_epoch_cadence_step", False)):
            epoch = int(state.get("epoch", 0))
            save_every = int(
                getattr(
                    cfg.train,
                    "save_every",
                    getattr(cfg.train, "save_every_epochs", self.save_every),
                )
            )
            if save_every > 0 and ((epoch + 1) % save_every) == 0:
                return

        self._save(
            cfg,
            state,
            next_epoch=int(state.get("epoch", 0)),
            version_label=f"step_{step:08d}",
        )

    def _save(
        self,
        cfg: Any,
        state: dict,
        *,
        next_epoch: int,
        version_label: str,
    ) -> None:
        step = int(state.get("global_step", 0))
        if step == self._last_save_step:
            return

        bundle: Optional[dict] = state.get("_ckpt_bundle", None)
        if bundle is None:
            raise RuntimeError(
                "CheckpointCallback requires state['_ckpt_bundle'] containing "
                "{model, optimizer, scheduler, scaler}."
            )

        model = bundle["model"]
        optimizer = bundle["optimizer"]
        scheduler = bundle.get("scheduler", None)
        scaler = bundle.get("scaler", None)
        masker_optimizer = bundle.get("masker_optimizer", None)
        masker_scheduler = bundle.get("masker_scheduler", None)

        ckpt_dir = str(getattr(cfg.train, "ckpt_dir", self.ckpt_dir))
        ckpt_name = str(getattr(cfg.train, "ckpt_name", self.ckpt_name))
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, ckpt_name)

        # Do NOT serialize private runtime keys.
        state_to_save = {k: v for k, v in state.items() if not str(k).startswith("_")}

        # Store the epoch that should run next. Epoch-end checkpoints move to
        # epoch + 1; mid-epoch step checkpoints restart the current epoch.
        state_to_save["next_epoch"] = int(next_epoch)

        ema_start = state.get(
            "_ema_start",
            float(getattr(cfg.model, "ema_momentum", [0.0])[0]),
        )
        if ema_start is not None:
            ema_start = float(ema_start)

        save_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            state=state_to_save,
            ema_start=ema_start,
            masker_optimizer=masker_optimizer,
            masker_scheduler=masker_scheduler,
        )

        # Versioned copy: keep epoch-numbered snapshots alongside last.pt.
        if bool(getattr(cfg.train, "keep_all_checkpoints", False)):
            versioned_path = os.path.join(ckpt_dir, f"{version_label}.pt")
            shutil.copy2(path, versioned_path)

        # Signal to the training loop that we saved, so it can trigger
        # callbacks.on_checkpoint_saved (e.g., for W&B artifact logging).
        state["_checkpoint_path"] = path
        self._last_save_step = step
