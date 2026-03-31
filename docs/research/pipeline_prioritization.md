# Pipeline Prioritization

This document should explain which dataset pipelines are worth building first and why.

## Goal

The project should not treat all candidate datasets equally. It should maintain an explicit priority order based on expected value and implementation cost.

## Prioritization Factors

Pipeline priority should consider:

- quality and trustworthiness of geometry
- overall data quality as a gating requirement
- likely contribution to the final unified corpus
- recurrence in strong modern model training recipes, when that evidence is available
- complementarity with already-supported datasets
- scale, once quality is acceptable
- camera-type diversity relative to the current dataset mix
- expected engineering effort
- expected maintenance burden
- upstream stability and accessibility

## Decision Rule

The prioritization should follow this order:

1. Quality is a hard constraint. Datasets that do not meet the quality bar should not be near-term priorities.
2. Among datasets that satisfy the quality bar, prefer larger datasets.
3. Among similarly strong candidates, prefer the ones that improve diversity in camera types and geometric conditions relative to the current mix.

Synthetic datasets are first-class targets under this rule. Real-captured datasets are also important, but neither category is automatically preferred if quality and strategic value point the other way.

Here, quality should be interpreted primarily as geometric quality. In particular, datasets with unreliable geometry or strong flying-pixel artifacts should be treated cautiously or de-prioritized even if they are otherwise attractive.

Camera diversity should be interpreted primarily as diversity in projective model and camera geometry, not only scene category.

Recurrence in strong modern model training recipes is a useful secondary signal, but it should not override the primary gate on geometric quality.

Practical convenience also should not override the primary gate on geometric quality. A dataset can be easy to ingest and still deserve low priority if its 3D quality is weak.

## Evidence From Recent Large-Scale Training

The MoGe 2 appendix provides a useful modern reference point for dataset value because it lists the datasets used in a large contemporary training mix and gives rough scale context.

That table should be interpreted carefully:

- recurrence in that training mix is a strong signal that a dataset is high-value
- the frame counts help contextualize scale across methods
- model quality does not necessarily correlate directly with the total amount of training data used
- inclusion in the table is evidence of strategic relevance, not automatic proof that the dataset should outrank the repository's geometry and implementation constraints

Datasets from that table that are especially relevant to this repository's roadmap include:

- `ARKitScenes`
- `MegaDepth`
- `Taskonomy`
- `Hypersim`
- `Structured3D`
- `TartanAir`
- `UrbanSyn`

Additional high-value datasets from that table that are not yet first-class targets here, but should be treated as strategically important references, include:

- `A2D2`
- `Argoverse2`
- `BlendedMVS`
- `ScanNet++`
- `Waymo`
- `ApolloSynthetic`
- `EDEN`
- `GTA-SfM`
- `IRS`
- `KenBurns`
- `MatrixCity`
- `MidAir`
- `MVS-Synth`
- `Synthia`
- `Synscapes`
- `UnrealStereo4K`
- `ObjaverseV1`

The practical implication is:

- every dataset named above should be treated as a high-value benchmark signal when evaluating future roadmap changes
- however, near-term integration order should still prefer the datasets whose geometry, packaging, and engineering cost fit the current repository best

## Minimum Readable Selection Path

For this repository, one practical prioritization factor is the size of the smallest source subset that still yields a readable `(image, distance, ray_dir)` sample.

This matters because the preferred workflow is:

- download the smallest complete useful unit
- run the full end-to-end pipeline on that unit
- scale up later without changing the pipeline shape

Based on the current dataset notes and HF triage in this repository, the high-value datasets with the most favorable known or likely minimum complete download units are:

- `UrbanSyn/UrbanSyn`
  - current implementation uses one aligned RGB + depth + semantic frame triplet as the complete unit
  - this is by far the smallest confirmed complete unit among the current high-value synthetic candidates
- `yaraalaa0/TopAir`
  - current implementation uses one trajectory folder such as `AssetsvilleTown_2`
  - the reviewed HF card indicates a complete trajectory folder is about `198 MB`, which is a favorable minimum-readable smoke unit
- `COLE-Ricoh/ToF-360`
  - current notes suggest one scene folder is the natural complete unit
  - only 4 scenes exist in the reviewed source, so the acquisition model looks manageable even though byte size is not yet pinned down
- `sayakpaul/nyu_depth_v2`
  - visible HF shard size is tiny
  - however, its geometry quality remains weak for this project, so low download cost does not make it a strong corpus target

High-value datasets whose minimum readable selection paths are currently known to be large, awkward, or not yet favorable for tiny smoke runs include:

- `Pointcept/arkitscenes-compressed`
  - one `arkitscenes_*.tar.gz` shard is about 9.5 GB on the reviewed HF tree
- `sayakpaul/diode-subset-train`
  - one `train_subset.tar.gz` is about 12.8 GB
- `Gen3DF/Structured3d-preprocessed`
  - current HF mirror appears to require effectively full-archive handling even for minimum-readable correctness
- `theairlabcmu/tartanair`
  - complete environment/difficulty/modality slices are operationally convenient, but still much heavier than per-frame datasets
- `GaussianWorld/Hypersim`
  - scene archives are a good abstraction boundary, but still materially heavier than per-frame or per-trajectory smoke units

Datasets whose minimum complete units remain strategically important but not yet well verified in this repository include:

- `Taskonomy`
- `MegaDepth`

For the specifically high-value datasets reinforced by the MoGe 2-style training evidence, the current acquisition-unit picture is:

- smallest confirmed complete unit:
  - `UrbanSyn/UrbanSyn`: one aligned frame triplet
- medium but still workable complete units:
  - `GaussianWorld/Hypersim`: one scene archive
  - `theairlabcmu/tartanair`: one complete environment/difficulty RGB-depth slice
- large or currently awkward minimum-readable paths:
  - `Pointcept/arkitscenes-compressed`: one archive shard, about 9.5 GB
  - `Gen3DF/Structured3d-preprocessed`: effectively very large archive-scale handling even for minimum-readable correctness
- high-value but minimum unit not yet well confirmed:
  - `MegaDepth`
  - `Taskonomy`

That means the current high-value datasets with relatively small or at least smoke-test-friendly complete units are:

1. `UrbanSyn/UrbanSyn`
2. `GaussianWorld/Hypersim`
3. `theairlabcmu/tartanair`

And the current high-value datasets with relatively unfavorable minimum complete units are:

1. `Pointcept/arkitscenes-compressed`
2. `Gen3DF/Structured3d-preprocessed`
3. `Taskonomy` / `MegaDepth` until their smallest complete units are confirmed

The practical reading is:

- if two datasets are similarly high-value, prefer the one with the smaller complete acquisition unit
- but do not let tiny download size override major geometry or strategic-value differences

## Priority Tiers

The roadmap can use simple tiers such as:

- `P0`: highest-value near-term targets
- `P1`: strong candidates after core abstractions are proven
- `P2`: speculative or lower-value additions

## Update Policy

This prioritization should be revised as:

- new datasets appear
- existing datasets disappear or degrade
- integration cost becomes clearer
- project goals shift

## Current Working Order

As of the current repository state:

- implemented high-priority synthetic pipelines:
  - `GaussianWorld/Hypersim`
  - `theairlabcmu/tartanair`
- implemented related or adjacent exploratory pipelines:
  - `theairlabcmu/TartanGround`
  - `UrbanSyn/UrbanSyn`
  - `sayakpaul/diode-subset-train`
  - `ZhengGuangze/VKITTI2_vlbm`
  - `ssbai/MegaDepth_v1`
  - `princeton-vl/WMGStereo`

The remaining priority tiers for new integration work are therefore:

- `P1`: `COLE-Ricoh/ToF-360`, `Gen3DF/Structured3d-preprocessed`
- `P2`: `sayakpaul/nyu_depth_v2`, `EnriqueSolarte/mp3d_fpe`

Strategic watchlist:

- `Taskonomy`

This list is intentionally provisional and should change as we inspect file layouts and actual geometric quality in more detail.

Current implementation-risk ordering inside the strongest remaining `P1` group:

1. `COLE-Ricoh/ToF-360`
2. `Gen3DF/Structured3d-preprocessed`

Interpretation note:

- `Hypersim` and `TartanAir` were the original early high-priority synthetic targets and are now already implemented.
- `TartanGround`, `UrbanSyn`, `TopAir`, and `Virtual KITTI 2` also exist in the repository, but their presence does not by itself reorder the remaining unimplemented `P1` targets beyond removing them from the unimplemented list.
- modern training-table recurrence, including the MoGe 2 dataset list above, strengthens the case that `ARKitScenes`, `Structured3D`, `MegaDepth`, `Hypersim`, `TartanAir`, and `UrbanSyn` are genuinely high-value datasets rather than incidental candidates.
- among the remaining unimplemented targets, `ToF-360` still adds useful camera-model diversity, but its live HF packaging proved materially heavier than expected. `Structured3D` remains strategically strong despite its heavy acquisition unit.

Roadmap changes should ideally be preceded by a documented triage pass rather than ad hoc promotion.
