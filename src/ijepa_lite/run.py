# FILE: src/ijepa_lite/run.py
from __future__ import annotations

import signal

import hydra
from omegaconf import DictConfig, OmegaConf

from ijepa_lite.build import build_for_task
from ijepa_lite.engine.eval_linear import linear_probe_eval
from ijepa_lite.engine.eval_suite import eval_suite
from ijepa_lite.engine.train_loop import train as pretrain_loop
from ijepa_lite.utils.dist import (
    barrier,
    cleanup_distributed,
    is_rank0,
    maybe_init_distributed,
    setup_device,
)
from ijepa_lite.utils.seed import set_seed

# ------------------------------------------------------------------
# Learned masker registration
#
# @register("name") only fires when the module is imported.  List all
# built-in learned maskers here so they are always available.  Add your
# own maskers below — one import per line is enough.
# ------------------------------------------------------------------
import ijepa_lite.masking.example_latent_masker   # noqa: F401  registers "gumbel_topk"
import ijepa_lite.masking.predictor_based_masker  # noqa: F401  registers "predictor_based"
import ijepa_lite.masking.rd_masker               # noqa: F401  registers "rd_3way"
# import my_project.my_masker                     # noqa: F401  registers "my_masker"


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)

    device = setup_device(cfg)
    maybe_init_distributed(cfg)

    # Synchronize start (safe when initialized)
    barrier(device)

    set_seed(int(cfg.seed))

    if is_rank0():
        safe_cfg = OmegaConf.to_container(cfg, resolve=False)
        print(OmegaConf.to_yaml(safe_cfg))

    def _handle_term(signum, frame):
        # Best-effort cleanup then raise to enter finally.
        cleanup_distributed(device)
        raise SystemExit(f"Received signal {signum}")

    try:
        # Register handlers only after dist init (so cleanup is meaningful).
        signal.signal(signal.SIGTERM, _handle_term)
        signal.signal(signal.SIGINT, _handle_term)

        bundle = build_for_task(cfg, device=device)

        task = str(cfg.task.name)
        if task == "pretrain":
            pretrain_loop(cfg, **bundle)
        elif task == "linear_probe":
            linear_probe_eval(cfg, **bundle)
        elif task == "eval_suite":
            eval_suite(cfg, **bundle)
        else:
            raise ValueError(f"Unknown task.name={task}")

    finally:
        cleanup_distributed(device)


if __name__ == "__main__":
    main()