from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from ijepa_lite.build import build_linear_probe_model
from ijepa_lite.data.datasets import (
    build_box_target_dataset,
    build_dataset,
    build_segmentation_dataset,
)
from ijepa_lite.data.transforms import (
    build_box_target_transforms,
    build_linear_probe_transforms,
    build_segmentation_transforms,
)
from ijepa_lite.utils.seed import set_seed


_IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    split: str | None


@dataclass(frozen=True)
class ClusterResult:
    method: str
    params: str
    labels: np.ndarray
    stats: dict[str, Any]


class ImageLabelCollate:
    def __call__(self, batch: list[Any]) -> dict[str, Any]:
        images = torch.stack([item[0] for item in batch], dim=0)
        labels = [item[1] for item in batch]
        return {"images": images, "labels": labels}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Visualize ViT patch-token embeddings from a pretrained I-JEPA "
            "checkpoint with PCA maps and multiple clustering methods."
        )
    )
    p.add_argument("--repo-root", default=".", help="Repository root containing configs/.")
    p.add_argument(
        "--data-root",
        required=True,
        help="Default dataset root passed to Hydra data.root.",
    )
    p.add_argument(
        "--dataset-root",
        action="append",
        default=None,
        help=(
            "Optional per-dataset root override in the form name=/path. "
            "Repeat when datasets live under different roots."
        ),
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path, or label=/path/to/checkpoint.pt.",
    )
    p.add_argument(
        "--dataset",
        action="append",
        required=True,
        help=(
            "Dataset name, optionally with split as name:split. "
            "Repeat for multiple datasets, e.g. --dataset stl10:test "
            "--dataset chestmnist:test."
        ),
    )
    p.add_argument("--out-dir", required=True, help="Directory for PNG/CSV/JSON outputs.")
    p.add_argument(
        "--experiment",
        default="downstream_mlp_earlystop",
        help="Hydra experiment used to build the encoder config.",
    )
    p.add_argument(
        "--encoder",
        default="auto",
        choices=("auto", "context", "target", "shared"),
        help="Checkpoint encoder subtree to use.",
    )
    p.add_argument(
        "--feature-mode",
        default="final",
        choices=("final", "last4_concat", "last4_mean"),
        help="Patch-token representation used for clustering.",
    )
    p.add_argument("--n-samples", type=int, default=8, help="Number of examples per dataset.")
    p.add_argument("--batch-size", type=int, default=16, help="Dataloader batch size.")
    p.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    p.add_argument(
        "--sample-mode",
        default="first",
        choices=("first", "random"),
        help="How to choose examples from each split.",
    )
    p.add_argument("--seed", type=int, default=0, help="Sampling and clustering seed.")
    p.add_argument(
        "--methods",
        nargs="+",
        default=("kmeans", "agglomerative", "spectral", "gmm", "dbscan"),
        choices=("kmeans", "agglomerative", "spectral", "gmm", "dbscan"),
        help="Clustering methods to run.",
    )
    p.add_argument(
        "--cluster-k",
        nargs="+",
        type=int,
        default=(2, 3, 4, 6, 8),
        help="Cluster counts for k-based methods.",
    )
    p.add_argument(
        "--dbscan-eps",
        nargs="+",
        type=float,
        default=(1.5, 2.5, 4.0),
        help="DBSCAN eps values in the standardized PCA cluster space.",
    )
    p.add_argument(
        "--cluster-pca-dim",
        type=int,
        default=16,
        help="PCA dimensions used for clustering after per-image standardization.",
    )
    p.add_argument(
        "--max-scatter-tokens",
        type=int,
        default=12000,
        help="Maximum token points to draw in global PCA scatter plots.",
    )
    p.add_argument("--model-arch", default=None, help="Override model.arch.")
    p.add_argument("--model-image-size", type=int, default=None, help="Override model.image_size.")
    p.add_argument("--model-patch-size", type=int, default=None, help="Override model.patch_size.")
    p.add_argument("--model-embed-dim", type=int, default=None, help="Override model.embed_dim.")
    p.add_argument("--model-num-heads", type=int, default=None, help="Override model.num_heads.")
    p.add_argument("--model-depth", type=int, default=None, help="Override model.depth.")
    p.add_argument(
        "--model-use-cls-token",
        default=None,
        choices=("true", "false"),
        help="Override model.use_cls_token.",
    )
    return p.parse_args()


def _parse_checkpoint(spec: str, repo_root: Path) -> tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
    else:
        raw = Path(spec)
        label = raw.stem
        raw_path = spec
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return label, path.resolve()


def _parse_dataset_spec(spec: str) -> DatasetSpec:
    if ":" in spec:
        name, split = spec.split(":", 1)
        return DatasetSpec(name=name, split=split)
    return DatasetSpec(name=spec, split=None)


def _parse_dataset_roots(specs: Iterable[str]) -> dict[str, str]:
    roots: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --dataset-root {spec!r}. Expected name=/path/to/root."
            )
        name, root = spec.split("=", 1)
        roots[name] = root
    return roots


def _compose_cfg(config_dir: Path, overrides: list[str]):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    return cfg


def _model_overrides(args: argparse.Namespace) -> list[str]:
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


def _dataset_cfg(
    *,
    config_dir: Path,
    args: argparse.Namespace,
    checkpoint_path: Path,
    dataset_name: str,
    data_root: str,
) -> Any:
    return _compose_cfg(
        config_dir,
        [
            f"experiment={args.experiment}",
            f"data={dataset_name}",
            f"data.root={data_root}",
            f"data.batch_size={args.batch_size}",
            f"data.num_workers={args.num_workers}",
            "data.download=false",
            "logger.mode=offline",
            f"task.pretrained_ckpt={checkpoint_path}",
            f"task.encoder={args.encoder}",
            *_model_overrides(args),
        ],
    )


def _default_split(cfg: Any) -> str:
    return str(getattr(cfg.data, "val_split", getattr(cfg.data, "train_split", "train")))


def _subset_indices(length: int, n_samples: int, mode: str, seed: int) -> list[int]:
    n = min(int(n_samples), int(length))
    if mode == "first":
        return list(range(n))
    rng = np.random.default_rng(seed)
    return sorted(int(x) for x in rng.choice(length, size=n, replace=False))


def _dataset_kind(dataset_name: str) -> str:
    name = str(dataset_name).lower()
    if name == "rsna_pneumonia_detection":
        return "box_target"
    if name in {"siimacr_pneumothorax", "vocseg"}:
        return "segmentation"
    return "classification"


def _build_viz_dataset(cfg: Any, split: str):
    kind = _dataset_kind(str(cfg.data.name))
    if kind == "box_target":
        _, val_tfm = build_box_target_transforms(cfg)
        return build_box_target_dataset(cfg.data, split=split, transforms=val_tfm)
    if kind == "segmentation":
        _, val_tfm = build_segmentation_transforms(cfg)
        return build_segmentation_dataset(cfg.data, split=split, transforms=val_tfm)
    _, val_tfm = build_linear_probe_transforms(cfg)
    return build_dataset(cfg.data, split=split, transform=val_tfm)


def _make_loader(dataset: Any, cfg: Any, indices: list[int]) -> DataLoader:
    subset = Subset(dataset, indices)
    num_workers = int(getattr(cfg.data, "num_workers", 4))
    return DataLoader(
        subset,
        batch_size=int(getattr(cfg.data, "batch_size", 16)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(getattr(cfg.data, "pin_memory", True)),
        persistent_workers=(
            bool(getattr(cfg.data, "persistent_workers", True)) if num_workers > 0 else False
        ),
        prefetch_factor=int(getattr(cfg.data, "prefetch_factor", 2)) if num_workers > 0 else None,
        collate_fn=ImageLabelCollate(),
    )


def _denormalize_image(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().float().permute(1, 2, 0).numpy()
    arr = arr * _IMAGENET_STD.reshape(1, 1, 3) + _IMAGENET_MEAN.reshape(1, 1, 3)
    return np.clip(arr, 0.0, 1.0)


def _label_summary(label: Any, class_names: list[str]) -> str:
    if isinstance(label, dict):
        boxes = label.get("boxes")
        n_boxes = int(boxes.shape[0]) if isinstance(boxes, torch.Tensor) else 0
        image_id = label.get("image_id")
        prefix = f"id={image_id}; " if image_id is not None else ""
        return f"{prefix}boxes={n_boxes}"
    if isinstance(label, (int, np.integer)):
        idx = int(label)
        if 0 <= idx < len(class_names):
            return class_names[idx]
        return str(idx)
    if not isinstance(label, torch.Tensor):
        return str(label)

    y = label.detach().cpu()
    if y.ndim >= 2:
        fg = float((y > 0).float().mean().item())
        return f"fg_fraction={fg:.4f}"
    if y.ndim == 0 or y.numel() == 1:
        idx = int(y.reshape(-1)[0].item())
        if 0 <= idx < len(class_names):
            return class_names[idx]
        return str(idx)
    active = torch.nonzero(y.reshape(-1) > 0.5, as_tuple=False).reshape(-1).tolist()
    if class_names:
        names = [class_names[idx] for idx in active if idx < len(class_names)]
        return ",".join(names) if names else "none"
    return ",".join(str(int(idx)) for idx in active) if active else "none"


@torch.no_grad()
def _encode_batch(
    encoder: torch.nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    feature_mode: str,
) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if feature_mode == "final":
        tokens = encoder(images)
    else:
        if not hasattr(encoder, "forward_last_n"):
            raise ValueError(f"feature_mode={feature_mode} requires encoder.forward_last_n.")
        layers = encoder.forward_last_n(images, last_n=4)
        if feature_mode == "last4_concat":
            tokens = torch.cat(layers, dim=-1)
        elif feature_mode == "last4_mean":
            tokens = torch.stack(layers, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unknown feature_mode={feature_mode}")
    return F.layer_norm(tokens, (tokens.shape[-1],)).detach().cpu()


def _grid_size(num_tokens: int) -> int:
    grid = int(math.isqrt(num_tokens))
    if grid * grid != num_tokens:
        raise ValueError(f"Expected square patch-token grid, got {num_tokens} tokens.")
    return grid


def _standardize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True)
    return (x - mu) / np.maximum(sigma, eps)


def _pca_projection(x: np.ndarray, n_components: int) -> tuple[np.ndarray, Any]:
    from sklearn.decomposition import PCA

    n = max(1, min(int(n_components), x.shape[0], x.shape[1]))
    pca = PCA(n_components=n, random_state=0)
    return pca.fit_transform(x), pca


def _token_pca_maps(tokens: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x = _standardize(tokens.astype(np.float32))
    proj, pca = _pca_projection(x, 3)
    if proj.shape[1] < 3:
        proj = np.pad(proj, ((0, 0), (0, 3 - proj.shape[1])), mode="constant")

    rgb = proj[:, :3]
    lo = np.percentile(rgb, 1, axis=0, keepdims=True)
    hi = np.percentile(rgb, 99, axis=0, keepdims=True)
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)

    pc1 = proj[:, 0]
    pc1 = (pc1 - pc1.min()) / max(float(pc1.max() - pc1.min()), 1e-6)

    stats = {
        "pca_explained_variance": [
            float(v) for v in getattr(pca, "explained_variance_ratio_", np.asarray([]))
        ],
    }
    return rgb.reshape(grid, grid, 3), pc1.reshape(grid, grid), stats


def _adjacent_cosine_stats(tokens: np.ndarray, grid: int) -> dict[str, float]:
    x = tokens.astype(np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    x = x.reshape(grid, grid, -1)

    distances: list[np.ndarray] = []
    for a, b in (
        (x[:-1, :, :], x[1:, :, :]),
        (x[:, :-1, :], x[:, 1:, :]),
    ):
        cosine = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
        distances.append((1.0 - cosine).reshape(-1))

    d = np.concatenate(distances)
    return {
        "adjacent_cosine_distance": float(d.mean()),
        "adjacent_cosine_distance_sd": float(d.std(ddof=1)) if d.size > 1 else 0.0,
        "adjacent_cosine_similarity": float(1.0 - d.mean()),
    }


def _cluster_space(tokens: np.ndarray, pca_dim: int) -> np.ndarray:
    x = _standardize(tokens.astype(np.float32))
    if pca_dim <= 0:
        return x
    proj, _ = _pca_projection(x, min(pca_dim, x.shape[0] - 1, x.shape[1]))
    return _standardize(proj.astype(np.float32))


def _connected_components(labels_2d: np.ndarray, cluster_label: int) -> int:
    mask = labels_2d == cluster_label
    if not mask.any():
        return 0

    seen = np.zeros_like(mask, dtype=bool)
    components = 0
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if seen[y, x] or not mask[y, x]:
                continue
            components += 1
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and mask[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return components


def _boundary_fraction(labels_2d: np.ndarray) -> float:
    edges = 0
    total = 0
    for axis in (0, 1):
        a = labels_2d.take(indices=range(labels_2d.shape[axis] - 1), axis=axis)
        b = labels_2d.take(indices=range(1, labels_2d.shape[axis]), axis=axis)
        valid = (a >= 0) & (b >= 0)
        total += int(valid.sum())
        edges += int(((a != b) & valid).sum())
    return float(edges / max(total, 1))


def _cluster_stats(labels: np.ndarray, x: np.ndarray, grid: int) -> dict[str, Any]:
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    labels = labels.astype(int)
    valid = labels >= 0
    unique = sorted(int(v) for v in np.unique(labels[valid]))
    counts = {str(k): int((labels == k).sum()) for k in unique}
    noise_count = int((labels < 0).sum())
    proportions = {
        str(k): float(v / max(int(valid.sum()), 1)) for k, v in counts.items()
    }
    p = np.asarray(list(proportions.values()), dtype=np.float64)
    entropy = float(-(p * np.log2(np.maximum(p, 1e-12))).sum()) if p.size else 0.0

    scores: dict[str, float | None] = {
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
    }
    if len(unique) > 1 and int(valid.sum()) > len(unique):
        xv = x[valid]
        yv = labels[valid]
        scores = {
            "silhouette": float(silhouette_score(xv, yv)),
            "calinski_harabasz": float(calinski_harabasz_score(xv, yv)),
            "davies_bouldin": float(davies_bouldin_score(xv, yv)),
        }

    labels_2d = labels.reshape(grid, grid)
    components = {
        str(k): int(_connected_components(labels_2d, k)) for k in unique
    }
    return {
        "n_clusters": int(len(unique)),
        "noise_count": noise_count,
        "counts": counts,
        "proportions": proportions,
        "cluster_entropy": entropy,
        "boundary_fraction": _boundary_fraction(labels_2d),
        "connected_components": components,
        **scores,
    }


def _run_clustering(
    x: np.ndarray,
    *,
    grid: int,
    methods: Iterable[str],
    cluster_k: Iterable[int],
    dbscan_eps: Iterable[float],
    seed: int,
) -> list[ClusterResult]:
    results: list[ClusterResult] = []

    for method in methods:
        if method == "kmeans":
            from sklearn.cluster import KMeans

            for k in cluster_k:
                if k <= 1 or k >= x.shape[0]:
                    continue
                model = KMeans(n_clusters=int(k), n_init=10, random_state=seed)
                labels = model.fit_predict(x)
                params = f"k={int(k)}"
                results.append(
                    ClusterResult(method, params, labels, _cluster_stats(labels, x, grid))
                )

        elif method == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering

            for k in cluster_k:
                if k <= 1 or k >= x.shape[0]:
                    continue
                model = AgglomerativeClustering(n_clusters=int(k), linkage="ward")
                labels = model.fit_predict(x)
                params = f"k={int(k)},linkage=ward"
                results.append(
                    ClusterResult(method, params, labels, _cluster_stats(labels, x, grid))
                )

        elif method == "spectral":
            from sklearn.cluster import SpectralClustering

            for k in cluster_k:
                if k <= 1 or k >= x.shape[0]:
                    continue
                neighbors = max(2, min(10, x.shape[0] - 1))
                model = SpectralClustering(
                    n_clusters=int(k),
                    affinity="nearest_neighbors",
                    n_neighbors=neighbors,
                    assign_labels="kmeans",
                    random_state=seed,
                )
                labels = model.fit_predict(x)
                params = f"k={int(k)},nn={neighbors}"
                results.append(
                    ClusterResult(method, params, labels, _cluster_stats(labels, x, grid))
                )

        elif method == "gmm":
            from sklearn.mixture import GaussianMixture

            for k in cluster_k:
                if k <= 1 or k >= x.shape[0]:
                    continue
                model = GaussianMixture(
                    n_components=int(k),
                    covariance_type="diag",
                    random_state=seed,
                )
                labels = model.fit_predict(x)
                params = f"k={int(k)},cov=diag"
                results.append(
                    ClusterResult(method, params, labels, _cluster_stats(labels, x, grid))
                )

        elif method == "dbscan":
            from sklearn.cluster import DBSCAN

            min_samples = max(3, int(round(math.sqrt(x.shape[0]) / 2)))
            for eps in dbscan_eps:
                model = DBSCAN(eps=float(eps), min_samples=min_samples)
                labels = model.fit_predict(x)
                params = f"eps={float(eps):g},min_samples={min_samples}"
                results.append(
                    ClusterResult(method, params, labels, _cluster_stats(labels, x, grid))
                )

    return results


def _render_label_map(ax: Any, labels: np.ndarray, grid: int, title: str) -> None:
    labels = labels.reshape(grid, grid).astype(float)
    ax.imshow(labels, interpolation="nearest", cmap="tab20")
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()


def _save_sample_figure(
    *,
    out_path: Path,
    image: np.ndarray,
    pca_rgb: np.ndarray,
    pc1: np.ndarray,
    clusters: list[ClusterResult],
    grid: int,
    title: str,
) -> None:
    plt = _pyplot()

    cluster_rows: list[list[ClusterResult]] = []
    for result in clusters:
        if not cluster_rows or cluster_rows[-1][0].method != result.method:
            cluster_rows.append([])
        if len(cluster_rows[-1]) == 3:
            cluster_rows.append([])
        cluster_rows[-1].append(result)

    cols = 3
    rows = 1 + len(cluster_rows)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), squeeze=False)

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("image", fontsize=8)
    axes[0, 0].set_axis_off()

    axes[0, 1].imshow(pca_rgb, interpolation="nearest")
    axes[0, 1].set_title("token PCA RGB", fontsize=8)
    axes[0, 1].set_axis_off()

    axes[0, 2].imshow(pc1, interpolation="nearest", cmap="magma")
    axes[0, 2].set_title("token PC1", fontsize=8)
    axes[0, 2].set_axis_off()

    for row_idx, row_clusters in enumerate(cluster_rows, start=1):
        for col_idx, result in enumerate(row_clusters):
            _render_label_map(
                axes[row_idx, col_idx],
                result.labels,
                grid,
                f"{result.method}\n{result.params}",
            )

    for ax in axes.reshape(-1):
        if ax.has_data():
            continue
        ax.set_axis_off()

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_global_pca_plots(
    *,
    out_dir: Path,
    records: list[dict[str, Any]],
    max_tokens: int,
    seed: int,
) -> None:
    if not records:
        return

    plt = _pyplot()

    tokens = np.concatenate([r["tokens"] for r in records], axis=0)
    dataset_ids: list[str] = []
    image_ids: list[str] = []
    for r in records:
        n = int(r["tokens"].shape[0])
        dataset_ids.extend([str(r["dataset"])] * n)
        image_ids.extend([str(r["sample_id"])] * n)

    if tokens.shape[0] > max_tokens:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(tokens.shape[0], size=max_tokens, replace=False))
        tokens = tokens[idx]
        dataset_ids = [dataset_ids[i] for i in idx]
        image_ids = [image_ids[i] for i in idx]

    proj, pca = _pca_projection(_standardize(tokens.astype(np.float32)), 2)
    meta = {
        "n_tokens_plotted": int(tokens.shape[0]),
        "explained_variance": [
            float(v) for v in getattr(pca, "explained_variance_ratio_", np.asarray([]))
        ],
    }
    (out_dir / "global_pca.json").write_text(json.dumps(meta, indent=2) + "\n")

    for color_by, values, path in (
        ("dataset", dataset_ids, out_dir / "global_token_pca_by_dataset.png"),
        ("image", image_ids, out_dir / "global_token_pca_by_image.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 6))
        unique = sorted(set(values))
        for value in unique:
            mask = np.asarray([v == value for v in values])
            ax.scatter(proj[mask, 0], proj[mask, 1], s=3, alpha=0.45, label=value)
        ax.set_title(f"Global token PCA colored by {color_by}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if len(unique) <= 20:
            ax.legend(markerscale=4, fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "checkpoint",
        "dataset",
        "split",
        "sample_id",
        "dataset_index",
        "label",
        "method",
        "params",
        "n_clusters",
        "noise_count",
        "cluster_entropy",
        "boundary_fraction",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "counts",
        "proportions",
        "connected_components",
        "pca_explained_variance",
        "adjacent_cosine_distance",
        "adjacent_cosine_distance_sd",
        "adjacent_cosine_similarity",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    repo_root = Path(args.repo_root).resolve()
    config_dir = repo_root / "configs"
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_label, ckpt_path = _parse_checkpoint(args.checkpoint, repo_root)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_specs = [_parse_dataset_spec(spec) for spec in args.dataset]
    if not dataset_specs:
        raise ValueError("At least one --dataset is required.")
    dataset_roots = _parse_dataset_roots(args.dataset_root or [])

    base_cfg = _dataset_cfg(
        config_dir=config_dir,
        args=args,
        checkpoint_path=ckpt_path,
        dataset_name=dataset_specs[0].name,
        data_root=dataset_roots.get(dataset_specs[0].name, args.data_root),
    )
    encoder = build_linear_probe_model(base_cfg).to(device)
    encoder.eval()

    all_rows: list[dict[str, Any]] = []
    all_token_records: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "checkpoint_label": ckpt_label,
        "checkpoint_path": str(ckpt_path),
        "feature_mode": str(args.feature_mode),
        "datasets": [],
        "outputs": {
            "summary_csv": "summary.csv",
            "summary_json": "summary.json",
            "global_pca_by_dataset": "global_token_pca_by_dataset.png",
            "global_pca_by_image": "global_token_pca_by_image.png",
        },
    }

    for spec in dataset_specs:
        cfg = _dataset_cfg(
            config_dir=config_dir,
            args=args,
            checkpoint_path=ckpt_path,
            dataset_name=spec.name,
            data_root=dataset_roots.get(spec.name, args.data_root),
        )
        split = spec.split or _default_split(cfg)
        full_ds = _build_viz_dataset(cfg, split)
        indices = _subset_indices(
            len(full_ds),
            n_samples=int(args.n_samples),
            mode=str(args.sample_mode),
            seed=int(args.seed),
        )
        loader = _make_loader(full_ds, cfg, indices)
        class_names = list(getattr(cfg.data, "class_names", []))

        dataset_out = out_dir / spec.name
        dataset_out.mkdir(parents=True, exist_ok=True)
        manifest["datasets"].append(
            {
                "name": spec.name,
                "split": split,
                "n_samples": len(indices),
                "indices": indices,
            }
        )

        seen = 0
        for batch in loader:
            images = batch["images"]
            labels = batch["labels"]
            tokens_b = _encode_batch(
                encoder,
                images,
                device=device,
                feature_mode=str(args.feature_mode),
            )
            for local_idx in range(images.shape[0]):
                dataset_index = indices[seen]
                sample_id = f"{spec.name}_{split}_{seen:03d}_idx{dataset_index}"
                tokens = tokens_b[local_idx].numpy().astype(np.float32)
                grid = _grid_size(tokens.shape[0])
                pca_rgb, pc1, pca_stats = _token_pca_maps(tokens, grid)
                continuous_stats = _adjacent_cosine_stats(tokens, grid)
                x_cluster = _cluster_space(tokens, int(args.cluster_pca_dim))
                clusters = _run_clustering(
                    x_cluster,
                    grid=grid,
                    methods=args.methods,
                    cluster_k=args.cluster_k,
                    dbscan_eps=args.dbscan_eps,
                    seed=int(args.seed),
                )
                label = _label_summary(labels[local_idx], class_names)

                _save_sample_figure(
                    out_path=dataset_out / f"{sample_id}.png",
                    image=_denormalize_image(images[local_idx]),
                    pca_rgb=pca_rgb,
                    pc1=pc1,
                    clusters=clusters,
                    grid=grid,
                    title=f"{ckpt_label} | {spec.name}:{split} | idx={dataset_index} | {label}",
                )

                sample_rows = []
                for result in clusters:
                    row = {
                        "checkpoint": ckpt_label,
                        "dataset": spec.name,
                        "split": split,
                        "sample_id": sample_id,
                        "dataset_index": int(dataset_index),
                        "label": label,
                        "method": result.method,
                        "params": result.params,
                        "pca_explained_variance": json.dumps(
                            pca_stats["pca_explained_variance"]
                        ),
                    }
                    row.update(continuous_stats)
                    row.update(
                        {
                            key: json.dumps(value) if isinstance(value, dict) else value
                            for key, value in result.stats.items()
                        }
                    )
                    sample_rows.append(row)

                all_rows.extend(sample_rows)
                all_token_records.append(
                    {
                        "dataset": spec.name,
                        "sample_id": sample_id,
                        "tokens": tokens,
                    }
                )
                seen += 1

    _write_summary_csv(out_dir / "summary.csv", all_rows)
    (out_dir / "summary.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _save_global_pca_plots(
        out_dir=out_dir,
        records=all_token_records,
        max_tokens=int(args.max_scatter_tokens),
        seed=int(args.seed),
    )

    print(f"[token_embedding_viz] wrote outputs to {out_dir}", flush=True)
    print(f"[token_embedding_viz] samples={len(all_token_records)} rows={len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()
