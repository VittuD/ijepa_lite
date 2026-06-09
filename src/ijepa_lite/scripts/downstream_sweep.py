from __future__ import annotations

import argparse
import gc
import os
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from ijepa_lite.build import (
    build_callbacks,
    build_detection_probe_loaders,
    build_linear_probe_loaders,
    build_linear_probe_model,
    build_segmentation_probe_loaders,
)
from ijepa_lite.engine.eval_detection import detection_probe_eval
from ijepa_lite.engine.eval_linear import linear_probe_eval
from ijepa_lite.engine.eval_segmentation import segmentation_probe_eval
from ijepa_lite.utils.dist import (
    barrier,
    cleanup_distributed,
    is_distributed,
    is_rank0,
    maybe_init_distributed,
    setup_device,
)
from ijepa_lite.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run a downstream probe sweep inside a single Python process. "
            "The encoder is loaded once per checkpoint and reused across datasets."
        )
    )
    p.add_argument("--repo-root", required=True, help="Absolute repo root.")
    p.add_argument("--data-root", required=True, help="Dataset root.")
    p.add_argument(
        "--output-root",
        default=None,
        help=(
            "Root directory for downstream run outputs. Defaults to "
            "<repo-root>/../outputs."
        ),
    )
    p.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint spec in the form label=/abs/or/rel/path/to/ckpt.pt",
    )
    p.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset name. Repeat to define the sweep order.",
    )
    p.add_argument(
        "--summary-path",
        default=None,
        help="Optional summary file to append progress/results to.",
    )
    p.add_argument(
        "--logger-mode",
        default="offline",
        help="Logger mode override passed into each composed cfg.",
    )
    p.add_argument(
        "--fairface-target-attr",
        default="race",
        help="FairFace target attr override when dataset=fairface.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running later sweep items even if one fails.",
    )
    p.add_argument(
        "--classification-experiment",
        default="downstream_mlp_earlystop",
        help="Experiment name to use for classification downstream tasks.",
    )
    p.add_argument(
        "--segmentation-experiment",
        default="downstream_vocseg_mlp_earlystop",
        help="Experiment name to use for segmentation downstream tasks.",
    )
    p.add_argument(
        "--detection-experiment",
        default="downstream_rsna_detection_mlp_earlystop",
        help="Experiment name to use for detection downstream tasks.",
    )
    p.add_argument(
        "--run-suffix",
        default="es_probe",
        help="Suffix appended to per-dataset run names.",
    )
    p.add_argument(
        "--task-pool",
        default=None,
        help="Optional override for task.pool, e.g. mean or last4_mean.",
    )
    p.add_argument("--model-arch", default=None, help="Override model.arch for the downstream encoder.")
    p.add_argument("--model-image-size", type=int, default=None, help="Override model.image_size.")
    p.add_argument("--model-patch-size", type=int, default=None, help="Override model.patch_size.")
    p.add_argument("--model-embed-dim", type=int, default=None, help="Override model.embed_dim.")
    p.add_argument("--model-num-heads", type=int, default=None, help="Override model.num_heads.")
    p.add_argument("--model-depth", type=int, default=None, help="Override model.depth.")
    p.add_argument(
        "--model-use-cls-token",
        default=None,
        choices=("true", "false"),
        help="Optional override for model.use_cls_token.",
    )
    p.add_argument(
        "--classification-batch-size",
        type=int,
        default=None,
        help="Optional data.batch_size override for classification downstream tasks.",
    )
    p.add_argument(
        "--segmentation-batch-size",
        type=int,
        default=None,
        help="Optional data.batch_size override for segmentation downstream tasks.",
    )
    p.add_argument(
        "--detection-batch-size",
        type=int,
        default=None,
        help="Optional data.batch_size override for detection downstream tasks.",
    )
    p.add_argument(
        "--classification-num-workers",
        type=int,
        default=None,
        help="Optional data.num_workers override for classification downstream tasks.",
    )
    p.add_argument(
        "--segmentation-num-workers",
        type=int,
        default=None,
        help="Optional data.num_workers override for segmentation downstream tasks.",
    )
    p.add_argument(
        "--detection-num-workers",
        type=int,
        default=None,
        help="Optional data.num_workers override for detection downstream tasks.",
    )
    p.add_argument(
        "--no-save-probe-checkpoints",
        action="store_true",
        help="Do not write probe head checkpoints; summaries and logs are still written.",
    )
    p.add_argument(
        "--train-epochs",
        type=int,
        default=None,
        help="Optional train.epochs override for all downstream tasks.",
    )
    p.add_argument(
        "--train-early-stop-patience",
        type=int,
        default=None,
        help="Optional train.early_stop_patience override for all downstream tasks.",
    )
    p.add_argument(
        "--train-early-stop-min-epochs",
        type=int,
        default=None,
        help="Optional train.early_stop_min_epochs override for all downstream tasks.",
    )
    return p.parse_args()


def _parse_checkpoint_specs(specs: Iterable[str], repo_root: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --checkpoint '{spec}'. Expected label=/path/to/checkpoint.pt"
            )
        label, raw_path = spec.split("=", 1)
        ckpt_path = Path(raw_path)
        if not ckpt_path.is_absolute():
            ckpt_path = repo_root / ckpt_path
        out.append((label, str(ckpt_path)))
    return out


def _compose_cfg(config_dir: Path, overrides: list[str]):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    return cfg


def _make_run_dir(output_root: Path, exp_name: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    if is_distributed():
        obj = [stamp] if is_rank0() else [None]
        dist.broadcast_object_list(obj, src=0)
        stamp = str(obj[0])
    day, clock = stamp.split("/", 1)
    run_dir = (
        output_root
        / day
        / f"{clock}_{exp_name}"
        / "rank0"
    )
    if is_rank0():
        run_dir.mkdir(parents=True, exist_ok=True)
        hydra_dir = run_dir / ".hydra"
        hydra_dir.mkdir(parents=True, exist_ok=True)
    barrier()
    return run_dir


def _write_hydra_snapshots(run_dir: Path, cfg, overrides: list[str]) -> None:
    if not is_rank0():
        barrier()
        return
    hydra_dir = run_dir / ".hydra"
    (hydra_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    (hydra_dir / "overrides.yaml").write_text("\n".join(overrides) + "\n")
    barrier()


def _append_summary(summary_path: Path | None, message: str) -> None:
    if not is_rank0():
        return
    stamped = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(stamped, flush=True)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("a") as f:
            f.write(stamped + "\n")


def _append_summary_block(summary_path: Path | None, header: str, lines: list[str]) -> None:
    if not is_rank0():
        return
    _append_summary(summary_path, header)
    for line in lines:
        print(line, flush=True)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("a") as f:
            for line in lines:
                f.write(line + "\n")


@contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _dataset_experiment(dataset: str, args: argparse.Namespace) -> str:
    if dataset in {"rsna_pneumonia_detection"}:
        return str(args.detection_experiment)
    if dataset in {"vocseg", "siimacr_pneumothorax"}:
        return str(args.segmentation_experiment)
    return str(args.classification_experiment)


def _model_overrides_from_args(args: argparse.Namespace) -> list[str]:
    overrides: list[str] = []
    if args.model_arch is not None:
        overrides.append(f"model.arch={args.model_arch}")
    if args.model_image_size is not None:
        overrides.append(f"model.image_size={args.model_image_size}")
    if args.model_patch_size is not None:
        overrides.append(f"model.patch_size={args.model_patch_size}")
    if args.model_embed_dim is not None:
        overrides.append(f"model.embed_dim={args.model_embed_dim}")
    if args.model_num_heads is not None:
        overrides.append(f"model.num_heads={args.model_num_heads}")
    if args.model_depth is not None:
        overrides.append(f"model.depth={args.model_depth}")
    if args.model_use_cls_token is not None:
        overrides.append(f"model.use_cls_token={args.model_use_cls_token}")
    return overrides


def _dataset_task(dataset: str) -> str:
    if dataset in {"rsna_pneumonia_detection"}:
        return "detection_probe"
    if dataset in {"vocseg", "siimacr_pneumothorax"}:
        return "segmentation_probe"
    return "linear_probe"


def _batch_overrides_for_dataset(
    dataset: str,
    *,
    classification_batch_size: int | None,
    segmentation_batch_size: int | None,
    detection_batch_size: int | None,
    classification_num_workers: int | None,
    segmentation_num_workers: int | None,
    detection_num_workers: int | None,
) -> list[str]:
    overrides: list[str] = []
    if _dataset_task(dataset) == "detection_probe":
        if detection_batch_size is not None:
            overrides.append(f"data.batch_size={detection_batch_size}")
        if detection_num_workers is not None:
            overrides.append(f"data.num_workers={detection_num_workers}")
        return overrides
    if _dataset_task(dataset) == "segmentation_probe":
        if segmentation_batch_size is not None:
            overrides.append(f"data.batch_size={segmentation_batch_size}")
        if segmentation_num_workers is not None:
            overrides.append(f"data.num_workers={segmentation_num_workers}")
        return overrides
    if classification_batch_size is not None:
        overrides.append(f"data.batch_size={classification_batch_size}")
    if classification_num_workers is not None:
        overrides.append(f"data.num_workers={classification_num_workers}")
    return overrides


def _build_run_overrides(
    *,
    dataset: str,
    data_root: str,
    ckpt_path: str,
    exp_name: str,
    logger_mode: str,
    fairface_target_attr: str,
    model_overrides: list[str],
    classification_batch_size: int | None,
    segmentation_batch_size: int | None,
    detection_batch_size: int | None,
    classification_num_workers: int | None,
    segmentation_num_workers: int | None,
    detection_num_workers: int | None,
    args: argparse.Namespace,
) -> list[str]:
    overrides = [
        f"experiment={_dataset_experiment(dataset, args)}",
        f"data={dataset}",
        f"data.root={data_root}",
        f"task.pretrained_ckpt={ckpt_path}",
        f"exp_name={exp_name}",
        f"logger.mode={logger_mode}",
    ]
    if args.task_pool is not None:
        overrides.append(f"task.pool={args.task_pool}")
    if args.no_save_probe_checkpoints:
        overrides.append("train.save_probe_checkpoints=false")
    if args.train_epochs is not None:
        overrides.append(f"train.epochs={args.train_epochs}")
    if args.train_early_stop_patience is not None:
        overrides.append(f"train.early_stop_patience={args.train_early_stop_patience}")
    if args.train_early_stop_min_epochs is not None:
        overrides.append(f"train.early_stop_min_epochs={args.train_early_stop_min_epochs}")
    overrides.extend(model_overrides)
    overrides.extend(
        _batch_overrides_for_dataset(
            dataset,
            classification_batch_size=classification_batch_size,
            segmentation_batch_size=segmentation_batch_size,
            detection_batch_size=detection_batch_size,
            classification_num_workers=classification_num_workers,
            segmentation_num_workers=segmentation_num_workers,
            detection_num_workers=detection_num_workers,
        )
    )
    if dataset == "fairface":
        overrides.append(f"data.target_attr={fairface_target_attr}")
    if dataset == "siimacr_pneumothorax":
        overrides.append("task.metric=fg_dice")
        overrides.append("task.foreground_class=1")
        overrides.append("task.loss=ce_dice")
        overrides.append("task.foreground_weight=20.0")
        overrides.append("task.ce_weight=1.0")
        overrides.append("task.dice_weight=1.0")
    return overrides


def _run_one_dataset(
    *,
    output_root: Path,
    config_dir: Path,
    encoder: torch.nn.Module,
    dataset: str,
    ckpt_label: str,
    ckpt_path: str,
    data_root: str,
    logger_mode: str,
    fairface_target_attr: str,
    model_overrides: list[str],
    classification_batch_size: int | None,
    segmentation_batch_size: int | None,
    detection_batch_size: int | None,
    classification_num_workers: int | None,
    segmentation_num_workers: int | None,
    detection_num_workers: int | None,
    args: argparse.Namespace,
) -> tuple[str, Path, dict[str, Any]]:
    exp_name = f"downstream_{ckpt_label}_{dataset}_{args.run_suffix}"
    overrides = _build_run_overrides(
        dataset=dataset,
        data_root=data_root,
        ckpt_path=ckpt_path,
        exp_name=exp_name,
        logger_mode=logger_mode,
        fairface_target_attr=fairface_target_attr,
        model_overrides=model_overrides,
        classification_batch_size=classification_batch_size,
        segmentation_batch_size=segmentation_batch_size,
        detection_batch_size=detection_batch_size,
        classification_num_workers=classification_num_workers,
        segmentation_num_workers=segmentation_num_workers,
        detection_num_workers=detection_num_workers,
        args=args,
    )
    cfg = _compose_cfg(config_dir, overrides)
    run_dir = _make_run_dir(output_root, exp_name)
    _write_hydra_snapshots(run_dir, cfg, overrides)

    # Reset RNG so each probe behaves like an independent one-off job.
    set_seed(int(cfg.seed))

    with _pushd(run_dir):
        callbacks = build_callbacks(cfg)
        task = _dataset_task(dataset)
        if task == "detection_probe":
            train_loader, val_loader, num_classes = build_detection_probe_loaders(cfg)
            metrics = detection_probe_eval(
                cfg=cfg,
                encoder=encoder,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=num_classes,
                callbacks=callbacks,
                device=next(encoder.parameters()).device,
            )
            del train_loader, val_loader, callbacks
        elif task == "segmentation_probe":
            train_loader, val_loader, num_classes = build_segmentation_probe_loaders(cfg)
            metrics = segmentation_probe_eval(
                cfg=cfg,
                encoder=encoder,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=num_classes,
                callbacks=callbacks,
                device=next(encoder.parameters()).device,
            )
            del train_loader, val_loader, callbacks
        else:
            train_loader, val_loader, num_classes = build_linear_probe_loaders(cfg)
            metrics = linear_probe_eval(
                cfg=cfg,
                encoder=encoder,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=num_classes,
                callbacks=callbacks,
                device=next(encoder.parameters()).device,
            )
            del train_loader, val_loader, callbacks
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return exp_name, run_dir, metrics


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):6.2f}"


def _truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _format_checkpoint_table(
    ckpt_label: str,
    rows: list[dict[str, Any]],
) -> list[str]:
    headers = ["dataset", "status", "metric", "train", "val", "best_val", "best_ep", "notes"]
    body: list[list[str]] = []
    for row in rows:
        dataset = str(row["dataset"])
        status = str(row["status"])
        if status == "ok":
            metrics = row["metrics"]
            if metrics["task_kind"] == "segmentation":
                metric_name = str(metrics.get("metric_name", "miou"))
                notes = (
                    f"pixel_acc={100.0 * float(metrics['val_acc']):.2f}; "
                    f"val_miou={100.0 * float(metrics['val_miou']):.2f}; "
                    f"best_val_miou={100.0 * float(metrics['best_val_miou']):.2f}; "
                    f"val_fg_iou={100.0 * float(metrics.get('val_fg_iou', 0.0)):.2f}; "
                    f"best_val_fg_iou={100.0 * float(metrics.get('best_val_fg_iou', 0.0)):.2f}; "
                    f"val_fg_dice={100.0 * float(metrics.get('val_fg_dice', 0.0)):.2f}; "
                    f"best_val_fg_dice={100.0 * float(metrics.get('best_val_fg_dice', 0.0)):.2f}"
                )
            elif metrics["task_kind"] == "detection":
                metric_name = str(metrics.get("metric_name", "map"))
                notes = (
                    f"val_map={100.0 * float(metrics.get('val_map', 0.0)):.2f}; "
                    f"best_val_map={100.0 * float(metrics.get('best_val_map', 0.0)):.2f}; "
                    f"val_ap50={100.0 * float(metrics.get('val_ap50', 0.0)):.2f}; "
                    f"best_val_ap50={100.0 * float(metrics.get('best_val_ap50', 0.0)):.2f}; "
                    f"val_froc={100.0 * float(metrics.get('val_froc_mean', 0.0)):.2f}; "
                    f"best_val_froc={100.0 * float(metrics.get('best_val_froc_mean', 0.0)):.2f}; "
                    f"image_auroc={100.0 * float(metrics.get('val_image_auroc', 0.0)):.2f}; "
                    f"num_gt={int(float(metrics.get('num_gt', 0.0)))}"
                )
            else:
                metric_name = str(metrics.get("metric_name", "acc1"))
                notes = ""
            body.append(
                [
                    dataset,
                    status,
                    metric_name,
                    _fmt_pct(metrics.get("train_metric", metrics.get("train_acc"))),
                    _fmt_pct(metrics.get("val_metric", metrics.get("val_acc"))),
                    _fmt_pct(metrics.get("best_val_metric", metrics.get("best_val_acc"))),
                    str(metrics.get("best_epoch", "-")),
                    notes,
                ]
            )
        else:
            body.append(
                [
                    dataset,
                    status,
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    str(row["error"]),
                ]
            )

    widths = [len(h) for h in headers]
    for row in body:
        for idx, cell in enumerate(row):
            limit = 72 if headers[idx] == "notes" else 20
            widths[idx] = max(widths[idx], min(len(cell), limit))

    def _render_row(row: list[str]) -> str:
        cells = []
        for idx, cell in enumerate(row):
            limit = 72 if headers[idx] == "notes" else 20
            text = _truncate(cell, min(widths[idx], limit))
            cells.append(text.ljust(min(widths[idx], limit)))
        return " | ".join(cells)

    sep = "-+-".join("-" * min(width, 72 if headers[idx] == "notes" else 20) for idx, width in enumerate(widths))
    lines = [
        f"Summary table for checkpoint: {ckpt_label}",
        _render_row(headers),
        sep,
    ]
    lines.extend(_render_row(row) for row in body)
    return lines


def main() -> None:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root is not None
        else (repo_root.parent / "outputs").resolve()
    )
    config_dir = repo_root / "configs"
    summary_path = (
        Path(args.summary_path).resolve() if args.summary_path is not None else None
    )
    data_root = str(Path(args.data_root))
    checkpoints = _parse_checkpoint_specs(args.checkpoint, repo_root)
    datasets = list(args.dataset)
    model_overrides = _model_overrides_from_args(args)
    bootstrap_cfg = _compose_cfg(
        config_dir,
        [
            f"experiment={args.classification_experiment}",
            f"data.root={data_root}",
            f"logger.mode={args.logger_mode}",
            *model_overrides,
        ],
    )
    device = setup_device(bootstrap_cfg)
    maybe_init_distributed(bootstrap_cfg)
    barrier(device)
    set_seed(int(bootstrap_cfg.seed))

    _append_summary(
        summary_path,
        (
            "starting downstream sweep "
            f"checkpoints={len(checkpoints)} datasets={len(datasets)} "
            f"device={device} world_size={dist.get_world_size() if is_distributed() else 1} "
            f"output_root={output_root}"
        ),
    )

    failures: list[tuple[str, str, str]] = []
    checkpoint_rows: dict[str, list[dict[str, Any]]] = {}

    try:
        for ckpt_label, ckpt_path in checkpoints:
            checkpoint_rows[ckpt_label] = []
            if not Path(ckpt_path).is_file():
                msg = f"checkpoint not found for {ckpt_label}: {ckpt_path}"
                _append_summary(summary_path, msg)
                failures.append((ckpt_label, "<checkpoint>", msg))
                checkpoint_rows[ckpt_label].append(
                    {"dataset": "<checkpoint>", "status": "fail", "error": msg}
                )
                if not args.continue_on_error:
                    break
                continue

            ckpt_cfg = _compose_cfg(
                config_dir,
                [
                    f"experiment={args.classification_experiment}",
                    f"task.pretrained_ckpt={ckpt_path}",
                    f"data.root={data_root}",
                    f"logger.mode={args.logger_mode}",
                    *model_overrides,
                ],
            )
            set_seed(int(ckpt_cfg.seed))
            encoder = build_linear_probe_model(ckpt_cfg).to(device)
            encoder.eval()
            encoder.requires_grad_(False)
            _append_summary(summary_path, f"loaded encoder ckpt={ckpt_label} path={ckpt_path}")

            for dataset in datasets:
                try:
                    _append_summary(summary_path, f"start ckpt={ckpt_label} dataset={dataset}")
                    exp_name, run_dir, metrics = _run_one_dataset(
                        output_root=output_root,
                        config_dir=config_dir,
                        encoder=encoder,
                        dataset=dataset,
                        ckpt_label=ckpt_label,
                        ckpt_path=ckpt_path,
                        data_root=data_root,
                        logger_mode=args.logger_mode,
                        fairface_target_attr=args.fairface_target_attr,
                        model_overrides=model_overrides,
                        classification_batch_size=args.classification_batch_size,
                        segmentation_batch_size=args.segmentation_batch_size,
                        detection_batch_size=args.detection_batch_size,
                        classification_num_workers=args.classification_num_workers,
                        segmentation_num_workers=args.segmentation_num_workers,
                        detection_num_workers=args.detection_num_workers,
                        args=args,
                    )
                    _append_summary(
                        summary_path,
                        f"ok ckpt={ckpt_label} dataset={dataset} exp={exp_name} run_dir={run_dir}",
                    )
                    checkpoint_rows[ckpt_label].append(
                        {"dataset": dataset, "status": "ok", "metrics": metrics}
                    )
                except Exception as exc:
                    tb = traceback.format_exc()
                    failures.append((ckpt_label, dataset, str(exc)))
                    checkpoint_rows[ckpt_label].append(
                        {"dataset": dataset, "status": "fail", "error": str(exc)}
                    )
                    _append_summary(
                        summary_path,
                        f"fail ckpt={ckpt_label} dataset={dataset} err={exc}\n{tb}",
                    )
                    if not args.continue_on_error:
                        raise

            del encoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            barrier(device)

        for ckpt_label, rows in checkpoint_rows.items():
            if not rows:
                continue
            _append_summary_block(
                summary_path,
                f"final table ckpt={ckpt_label}",
                _format_checkpoint_table(ckpt_label, rows),
            )

        if failures:
            lines = "; ".join(
                f"{ckpt}/{dataset}: {msg}" for ckpt, dataset, msg in failures
            )
            raise SystemExit(f"downstream sweep finished with failures: {lines}")

        _append_summary(summary_path, "downstream sweep finished successfully")
    finally:
        cleanup_distributed(device)


if __name__ == "__main__":
    main()
