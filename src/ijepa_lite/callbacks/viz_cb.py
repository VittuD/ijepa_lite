from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from ijepa_lite.callbacks.base import Callback
from ijepa_lite.utils.dist import is_rank0, unwrap_model


class VizCallback(Callback):
    """
    Opt-in visualization of Goldilocks masker scoring.

    Activated when ``cfg.train.viz_enabled`` is True.  Rank 0 only.

    Cadence: ``cfg.train.viz_every_epochs`` (alias ``viz_every``).  When not
    provided, falls back to ``save_every`` / ``save_every_epochs`` so the
    default behaviour is unchanged (viz piggybacks on checkpoint saving).

    Only activates if the model has a ``latent_masker`` with a
    ``target_soft``-returning interface (e.g. GoldilocksTeacherMasker).
    """

    def __init__(self) -> None:
        self._enabled: bool = False
        self._viz_every: int = 1
        self._dataset = None
        self._cfg_viz: Any = None
        self._image_size: int = 96
        self._patch_size: int = 8
        self._is_multiblock: bool = False
        self._collateMasker = None

    def on_run_start(self, cfg: Any, state: dict, model: Any) -> None:
        self._enabled = bool(getattr(cfg.train, "viz_enabled", False))
        if not self._enabled or not is_rank0():
            self._enabled = False
            return

        # Check that model has a compatible latent masker
        core = model
        masker = getattr(core, "latent_masker", None)
        is_multiblock = (masker is None and
                         getattr(cfg.masking, "name", None) == "multiblock")
        if masker is None and not is_multiblock:
            print("[VizCallback] No latent_masker found — disabling.")
            self._enabled = False
            return
        self._is_multiblock = is_multiblock
        if is_multiblock:
            from ijepa_lite.build import _build_collate_masker
            self._collateMasker = _build_collate_masker(cfg)
        elif not hasattr(core, "get_viz_encoder"):
            print("[VizCallback] Model does not expose get_viz_encoder() — disabling.")
            self._enabled = False
            return

        save_every = int(
            getattr(cfg.train, "save_every",
                    getattr(cfg.train, "save_every_epochs", 1))
        )
        self._viz_every = int(
            getattr(cfg.train, "viz_every",
                    getattr(cfg.train, "viz_every_epochs", save_every))
        )
        self._image_size = int(cfg.model.image_size)
        self._patch_size = int(cfg.model.patch_size)
        self._cfg_viz = getattr(cfg.train, "viz", None)

        # Pre-load the dataset (raw, no transforms — viz handles its own)
        vcfg = self._cfg_viz
        data_root = str(getattr(vcfg, "data_root", "/scratch/datasets/"))
        dataset_name = str(getattr(vcfg, "dataset", "stl10"))

        try:
            from torchvision import datasets as tv_datasets
            if dataset_name == "stl10":
                self._dataset = tv_datasets.STL10(data_root, split="test", download=False)
            elif dataset_name == "food101":
                self._dataset = tv_datasets.Food101(data_root, split="test", download=False)
            elif dataset_name == "chestmnist":
                from medmnist import ChestMNIST
                self._dataset = ChestMNIST(
                    root=data_root,
                    split=str(getattr(vcfg, "split", "test")),
                    size=int(getattr(vcfg, "size", 224)),
                    as_rgb=True,
                    download=False,
                )
                self._dataset.classes = list(
                    getattr(
                        vcfg,
                        "class_names",
                        [
                            "atelectasis",
                            "cardiomegaly",
                            "effusion",
                            "infiltration",
                            "mass",
                            "nodule",
                            "pneumonia",
                            "pneumothorax",
                            "consolidation",
                            "edema",
                            "emphysema",
                            "fibrosis",
                            "pleural",
                            "hernia",
                        ],
                    )
                )
            else:
                print(f"[VizCallback] Unknown dataset '{dataset_name}' — disabling.")
                self._enabled = False
                return
        except Exception as e:
            print(f"[VizCallback] Could not load dataset: {e} — disabling.")
            self._enabled = False
            return

    def on_epoch_end(self, cfg: Any, state: dict, metrics: Dict[str, float]) -> None:
        if not self._enabled or not is_rank0():
            return

        epoch = int(state.get("epoch", 0))
        if self._viz_every <= 0:
            return
        if (epoch + 1) % self._viz_every != 0:
            return

        if not self._is_multiblock:
            bundle = state.get("_ckpt_bundle")
            if bundle is not None:
                core = unwrap_model(bundle["model"])
                masker = getattr(core, "latent_masker", None)
                if (
                    masker is not None
                    and bool(getattr(masker, "warmup_use_vanilla_multiblock", False))
                    and epoch < int(getattr(masker, "warmup_epochs", 0))
                ):
                    print(
                        "[VizCallback] skipping visualization during "
                        f"vanilla-multiblock warmup (epoch={epoch}, "
                        f"warmup_epochs={int(getattr(masker, 'warmup_epochs', 0))})."
                    )
                    return

        self._run_viz(cfg, state, epoch)

    def _run_viz(self, cfg: Any, state: dict, epoch: int) -> None:
        import torch

        vcfg = self._cfg_viz
        n_images = int(getattr(vcfg, "n_images", 100))
        out_base = str(getattr(vcfg, "out_dir", "viz_output"))
        grid_cols = int(getattr(vcfg, "grid_cols", 10))

        out_dir = Path(out_base) / f"epoch_{epoch:05d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        dataset_name = str(getattr(vcfg, "dataset", "stl10"))
        class_names = []
        if hasattr(self._dataset, "classes"):
            class_names = list(self._dataset.classes)

        if self._is_multiblock:
            from ijepa_lite.viz.goldilocks_viz import (
                save_avg_coverage_heatmap,
                save_class_coverage_heatmap,
                visualize_split_multiblock,
            )

            sums, cls_sums, cls_n = visualize_split_multiblock(
                dataset_name=dataset_name,
                split="test",
                dataset=self._dataset,
                masker=self._collateMasker,
                out_dir=out_dir,
                n=n_images,
                grid_cols=grid_cols,
                patch_size=self._patch_size,
                image_size=self._image_size,
                verbose=False,
            )

            save_avg_coverage_heatmap(
                sums, n_images,
                out_dir / f"{dataset_name}_avg_coverage.png",
                verbose=False,
            )
            if cls_sums:
                save_class_coverage_heatmap(
                    sums, n_images,
                    cls_sums, cls_n,
                    class_names,
                    out_dir / f"{dataset_name}_per_class_coverage.png",
                    verbose=False,
                )
            print(f"[VizCallback] epoch={epoch}  output -> {out_dir}/")
            return

        from ijepa_lite.viz.goldilocks_viz import (
            save_avg_3way_heatmap,
            save_avg_nway_heatmap,
            save_avg_score_heatmap,
            save_class_3way_heatmap,
            save_class_nway_heatmap,
            save_class_score_heatmap,
            visualize_split,
        )

        k_tgt = getattr(vcfg, "k_tgt", None)
        if k_tgt is not None:
            k_tgt = int(k_tgt)

        bundle = state.get("_ckpt_bundle")
        if bundle is None:
            return
        model = bundle["model"]
        core = unwrap_model(model)
        encoder = core.get_viz_encoder()
        masker = core.latent_masker

        if masker is None:
            return

        device = next(encoder.parameters()).device
        encoder.eval()
        masker.eval()

        sums, cls_sums, cls_n, viz_type = visualize_split(
            dataset_name=dataset_name,
            split="test",
            dataset=self._dataset,
            encoder=encoder,
            masker=masker,
            device=device,
            out_dir=out_dir,
            n=n_images,
            grid_cols=grid_cols,
            patch_size=self._patch_size,
            image_size=self._image_size,
            k_tgt=k_tgt,
            epoch=epoch,
            verbose=False,
        )

        if viz_type == "nway":
            # Infer M from role_sums keys
            M = sum(1 for k in sums if k.startswith("tgt_"))
            save_avg_nway_heatmap(
                sums, n_images,
                out_dir / f"{dataset_name}_avg_nway.png",
                num_tgt_blocks=M,
                verbose=False,
            )
            if cls_sums:
                save_class_nway_heatmap(
                    sums, n_images,
                    cls_sums, cls_n,
                    class_names,
                    out_dir / f"{dataset_name}_per_class_nway.png",
                    num_tgt_blocks=M,
                    verbose=False,
                )
        elif viz_type == "3way" or viz_type is True:
            save_avg_3way_heatmap(
                sums, n_images,
                out_dir / f"{dataset_name}_avg_3way.png",
                verbose=False,
            )
            if cls_sums:
                save_class_3way_heatmap(
                    sums, n_images,
                    cls_sums, cls_n,
                    class_names,
                    out_dir / f"{dataset_name}_per_class_3way.png",
                    verbose=False,
                )
        else:
            save_avg_score_heatmap(
                sums, n_images,
                out_dir / f"{dataset_name}_avg_score.png",
                verbose=False,
            )
            if cls_sums:
                save_class_score_heatmap(
                    sums, n_images,
                    cls_sums, cls_n,
                    class_names,
                    out_dir / f"{dataset_name}_per_class_score.png",
                    verbose=False,
                )

        encoder.train()
        masker.train()
        print(f"[VizCallback] epoch={epoch}  output -> {out_dir}/")
