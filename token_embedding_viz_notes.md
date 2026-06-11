# Token Embedding Visualization Notes

This note summarizes how to read the token embedding visualizations used for the task-relevance masking runs.

## What the script visualizes

The script is `src/ijepa_lite/scripts/visualize_token_embeddings.py`.

For each checkpoint and image, it:

1. Loads a frozen ViT encoder from `task.pretrained_ckpt`.
2. Applies the validation transform at the requested `model.image_size`.
3. Runs the image through the encoder and gets patch-token embeddings.
4. Layer-normalizes each token embedding.
5. Runs per-image PCA and clustering on those normalized patch tokens.

So the visualizations act on patch-token embeddings, not raw pixels.

At ViT-H/14:

- `224x224` gives a `16x16` grid, or `256` patch tokens.
- `448x448` gives a `32x32` grid, or `1024` patch tokens.

The `448x448` case uses positional embedding interpolation from the original checkpoint grid. It is useful as a qualitative stress test, but the `224x224` case is the cleaner match to the downstream probe geometry.

## RGB PCA

For each image independently, the script takes:

```text
tokens: [num_patches, embed_dim]
```

It standardizes each feature dimension across patches, fits PCA with 3 components, and maps PC1/PC2/PC3 to RGB. Each channel is rescaled by its 1st and 99th percentiles.

Important consequence:

- RGB colors are relative within one image.
- The same color in two different images or checkpoints is not guaranteed to be the same semantic direction.
- What matters is the spatial organization: smooth fields, anatomical regions, fragmented islands, and isolated outlier tokens.

A strongly different-colored isolated patch means that token is far from the rest of that image's token cloud along one or more top PCA directions.

## Token PC1

Token PC1 is the first component from the same per-image PCA used for RGB PCA. It is min-max scaled and shown as a single heatmap.

It shows the dominant axis of variation among patch tokens in that image. On chest X-rays, this may align with broad anatomy, lung/background contrast, intensity structure, pathology-related texture, or nuisance/global-token behavior. The visualization itself does not identify which one.

## KMeans And GMM

KMeans and GMM do not act on pixels or on the RGB PCA image.

They act on a clustering space built from the same normalized patch tokens:

1. Standardize token features across patches.
2. Fit PCA down to `cluster_pca_dim` dimensions, default `16`.
3. Standardize the PCA features again.
4. Run the clustering method per image.

KMeans partitions the token cloud by nearest centroids in this PCA-reduced token space. GMM fits diagonal-covariance Gaussian components in the same space and renders the most likely component as a hard label.

GMM is useful as a sanity check, but it is not the main evidence. It ignores spatial adjacency, assumes Gaussian-like components, and can make scattered tokens look like a legitimate component if they share feature statistics.

## Quantifying Smoothness

The most useful quantitative smoothness signals are already in `summary.csv`.

Use these per image, method, and `k`:

- `boundary_fraction`: fraction of neighboring patch pairs with different cluster labels. Lower means smoother spatial fields.
- `connected_components`: number of disconnected islands per cluster. Fewer components means less fragmentation.
- `adjacent_cosine_distance`: average `1 - cosine_similarity` over horizontal
  and vertical neighboring patch-token pairs before clustering. Lower means the
  continuous token field is smoother, independent of k-means discretization.
- `cluster_entropy`: balance of cluster sizes. Useful, but not a smoothness
  metric by itself.

A good aggregate is:

```text
mean_boundary_fraction
mean_total_connected_components = sum connected_components over clusters
mean_extra_components = sum(max(components - 1, 0)) over clusters
mean_adjacent_cosine_distance
```

Compare these on the same images, same `k`, same method, and same image size.
`adjacent_cosine_distance` does not depend on `k` or clustering method, but it
should still be compared on the same images, checkpoint feature mode, and patch
grid.

If affinity-novelty has lower boundary fraction and fewer connected components than vanilla while silhouette/Calinski/Davies-Bouldin are similar, that means the cluster separability in feature space is not dramatically different, but the spatial arrangement of the token field is smoother.

If the same comparison also has lower `adjacent_cosine_distance`, the smoothness
claim is stronger because it holds in the continuous token field and not only in
the induced k-means label map.

For compact ranking, use the per-method files from
`scripts/analyze_task_relevance_token_viz_groups.py`, especially:

```text
stats_by_family/all/rankings_by_method/kmeans_ranking_by_checkpoint.md
stats_by_family/all/rankings_by_method/gmm_ranking_by_checkpoint.md
```

Do not treat pooled KMeans+GMM rankings as the main comparison. Each clustering method has different assumptions, so the fair compact table is one ranking per method.

## Reading The Current Pattern

The qualitative pattern observed so far is:

- Vanilla multiblock often has isolated outlier-like patch tokens in RGB PCA.
- Affinity-novelty has much smoother maps and many fewer token islands.
- This appears in both IN1K continuation and ChestMNIST continuation, stronger in the chest setting.

A plausible interpretation is:

```text
Vanilla remains a strong performance baseline, but permits some patch tokens to act like global accumulator or sink tokens. Affinity-novelty redistributes information more evenly over the spatial token field, reducing isolated outlier-token artifacts.
```

This is not automatically a downstream-performance win. If downstream scores are tied within roughly +/-1%, then the honest claim is about representation organization:

```text
Affinity-novelty changes the spatial geometry of patch-token embeddings without materially changing downstream separability under the current probes.
```

That is still useful. It suggests token-field smoothness is an orthogonal axis of representation quality: potentially relevant for interpretability, robustness, open-vocabulary localization, few-shot dense adaptation, or tasks that depend on spatially faithful token-level reasoning.

## Current 448 KMeans Interpretation

The 448x448 KMeans ranking over all visualized datasets supports the qualitative observation that affinity-novelty produces the least fragmented token fields. This is stronger than a single anecdotal image: the effect survives averaging over the visualized datasets and samples.

The main pattern is that affinity-novelty is smoother than vanilla in both the ChestMNIST-continuation and IN1K-continuation families. That suggests the effect is not merely a chest-domain artifact. It looks like the masker changes a more general property of the representation: how patch-token information is distributed over the spatial grid.

The effect is still stronger in the ChestMNIST-continuation family. A plausible interpretation is that chest X-rays provide a more spatially regular image domain, so the same affinity-novelty pressure has a clearer visual expression: token neighborhoods become more anatomy-like and less salt-and-pepper.

The ranking also suggests that some HMarg-style variants, especially in the IN1K-continuation family, can make the token field more fragmented. This does not necessarily mean those representations are useless, but it suggests their training pressure may encourage less spatially coherent patch-token geometry under this visualization probe.

The PCA explained-variance pattern does not point to a simple "smoother means lower-rank" explanation. The smoother affinity-novelty maps do not appear to be just washed-out or collapsed token clouds. The relevant change is spatial organization: fewer fragmented islands and fewer outlier-like token regions, while the broad PCA variance scale remains comparable.

The current best interpretation is:

```text
Affinity-novelty appears to regularize the spatial organization of patch-token embeddings. It reduces fragmentation and outlier-token islands without simply collapsing the token cloud or making the PCA spectrum trivially smaller.
```

This should be stated as a representation-geometry result, not as proof of better downstream task utility.

## Main Caveat

Pretty PCA maps are not proof of better medical representation quality. They support a mechanism hypothesis. The primary evidence for utility remains the downstream metrics; the maps explain how two performance-equivalent representations may organize information differently.
