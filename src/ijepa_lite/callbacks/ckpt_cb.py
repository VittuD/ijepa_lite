from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ijepa_lite.callbacks.base import Callback
from ijepa_lite.engine.checkpoint import save_checkpoint
from ijepa_lite.utils.dist import is_rank0


@dataclass
class CheckpointCallback(Callback):
    """
    Saves training checkpoints on a fixed epoch cadence.

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

        ckpt_dir = str(getattr(cfg.train, "ckpt_dir", self.ckpt_dir))
        ckpt_name = str(getattr(cfg.train, "ckpt_name", self.ckpt_name))
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, ckpt_name)

        # Do NOT serialize private runtime keys.
        state_to_save = {k: v for k, v in state.items() if not str(k).startswith("_")}

        # Store the epoch that should run *next* so resume doesn't re-run the
        # last completed epoch (off-by-one fix).
        state_to_save["next_epoch"] = epoch + 1

        ema_start = float(
            state.get("_ema_start", float(getattr(cfg.model, "ema_momentum", [0.0])[0]))
        )

        save_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            state=state_to_save,
            ema_start=ema_start,
        )

        # Signal to the training loop that we saved, so it can trigger
        # callbacks.on_checkpoint_saved (e.g., for W&B artifact logging).
        state["_checkpoint_path"] = path
