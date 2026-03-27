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

## Initial Working Order

As of 2026-03-27, the first-pass working order is:

- `P0`: `GaussianWorld/Hypersim`
- `P1`: `theairlabcmu/tartanair`, `Pointcept/arkitscenes-compressed`, `Gen3DF/Structured3d-preprocessed`, `sayakpaul/diode-subset-train`, `COLE-Ricoh/ToF-360`, `UrbanSyn/UrbanSyn`, `yaraalaa0/TopAir`
- `P2`: `sayakpaul/nyu_depth_v2`, `EnriqueSolarte/mp3d_fpe`, `idsia-robotics/micro-tartanair`

Strategic watchlist:

- `Taskonomy`
- `Virtual KITTI 2`

This list is intentionally provisional and should change as we inspect file layouts and actual geometric quality in more detail.

Current implementation-risk ordering inside the strongest `P1` group:

1. `theairlabcmu/tartanair`
2. `Pointcept/arkitscenes-compressed`
3. `Gen3DF/Structured3d-preprocessed`

Roadmap changes should ideally be preceded by a documented triage pass rather than ad hoc promotion.
