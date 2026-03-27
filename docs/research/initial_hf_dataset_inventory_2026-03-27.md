# Initial Hugging Face Dataset Inventory

Date of review: 2026-03-27

This is a first-pass inventory of candidate datasets visible on Hugging Face and relevant to the current project direction.

It is intentionally practical rather than exhaustive. The goal is to identify promising pipeline targets and reject obviously poor fits early.

## Interpretation Notes

- availability was checked on Hugging Face on 2026-03-27
- quality judgments below are preliminary and partly inferential
- geometric quality is weighted more heavily than image quality
- severe risk of spurious 3D points between objects should be treated as a major negative
- indoor-first strategy affects near-term priority
- camera diversity refers mainly to projection and camera-model diversity

## Shortlist

| Dataset | HF status | Domain | Projection / camera notes | Scale signal | Geometry signal | Integration friction | Preliminary value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sayakpaul/nyu_depth_v2` | accessible | indoor real | Kinect-style perspective RGB-D | small | legacy and convenient, but weak for geometry quality | low | limited/debug value |
| `GaussianWorld/Hypersim` | gated, no card | indoor synthetic | perspective indoor renderings | large | likely very strong | medium to high | very high |
| `theairlabcmu/tartanair` | accessible, very large HF package | mixed synthetic | perspective stereo / sequence data | very large | likely very strong | medium to high | very high |
| `Pointcept/arkitscenes-compressed` | accessible, large compressed package | indoor real | perspective RGB-D with device calibration likely preserved if packaging is faithful | large | likely strong | high | very high |
| `Gen3DF/Structured3d-preprocessed` | accessible, very large chunked archive | indoor synthetic | likely perspective and panoramic structured renders | very large | likely very strong | high | very high |
| `sayakpaul/diode-subset-train` | accessible subset, rough packaging | indoor + outdoor real | perspective RGB-D | medium subset | likely strong | medium | high but packaging risk |
| `COLE-Ricoh/ToF-360` | accessible | indoor real | equirectangular / panoramic RGB-D | small | promising and distinctive | medium | strategically high |
| `EnriqueSolarte/mp3d_fpe` | gated | indoor 360 | equirectangular panorama depth | large storage, unclear immediate fit | promising | high | high for projection diversity |
| `UrbanSyn/UrbanSyn` | accessible | outdoor street synthetic | perspective driving | medium to large | likely strong synthetic geometry | medium | high later |
| `yaraalaa0/TopAir` | accessible | aerial synthetic | nadir aerial perspective | medium | likely strong synthetic geometry | low to medium | good diversification target |
| `idsia-robotics/micro-tartanair` | accessible | synthetic mixed navigation | perspective, but tiny 48x48 export | very large row count, tiny resolution | synthetic GT but reduced utility | low | low for main corpus |

## Dataset Notes

### `sayakpaul/nyu_depth_v2`

- Why it matters:
  - indoor-first
  - historically important RGB-D dataset
  - likely low-friction first real-data pipeline
- Concerns:
  - relatively small
  - Hugging Face card indicates preprocessed filled depth rather than raw depth only
  - known risk of flying 3D points / unreliable geometry for this project
  - likely less useful for large-scale ambition than newer synthetic datasets
- Initial judgment:
  - useful only as a convenience or debugging target
  - should not be treated as a top-priority geometry dataset

### `GaussianWorld/Hypersim`

- Why it matters:
  - strong indoor fit
  - synthetic data with dense ground-truth geometry is aligned with the geometry-first philosophy
  - much better scale than NYU-like datasets
- Concerns:
  - Hugging Face packaging is gated and has no dataset card
  - integration details will need inspection after access
- Initial judgment:
  - one of the most important indoor targets if the HF packaging is workable

### `sayakpaul/diode-subset-train`

- Why it matters:
  - DIODE is known for accurate dense long-range indoor and outdoor depth from one sensor suite
  - useful bridge between indoor and outdoor real data
- Concerns:
  - current HF artifact is only a subset
  - viewer and split parsing look broken on Hugging Face
  - packaging quality may slow implementation
- Initial judgment:
  - high-interest candidate, but not ideal as the very first pipeline

### `theairlabcmu/tartanair`

- Why it matters:
  - very large synthetic source
  - recurrent in several strong modern geometry/depth recipes
  - likely valuable for both scale and camera-motion diversity
- Concerns:
  - extremely large HF package
  - packaging is archive-heavy and operationally expensive
  - not as low-friction as a smaller indoor bootstrap dataset
- Initial judgment:
  - strong `P1` target and possibly one of the highest-value scale candidates after `Hypersim`

### `Pointcept/arkitscenes-compressed`

- Why it matters:
  - large real indoor RGB-D source
  - reinforced by recurrent appearance in strong geometry-model recipes
  - higher long-term value than tiny legacy RGB-D sets if packaging is usable
- Concerns:
  - current HF package is compressed/processed and not yet inspected internally
  - calibration preservation must be verified
- Initial judgment:
  - strong `P1` target for real indoor data

### `Gen3DF/Structured3d-preprocessed`

- Why it matters:
  - large synthetic indoor source
  - reinforced by repeated use in modern geometry-model mixtures
  - strong complement to `Hypersim`
- Concerns:
  - extremely large package
  - chunked raw-style archive increases operational friction
- Initial judgment:
  - strong `P1` target after top bootstrap paths are stable

### `COLE-Ricoh/ToF-360`

- Why it matters:
  - unusual and strategically valuable projection type
  - explicit spherical RGB-D setup fits the long-term camera-diversity goal well
  - indoor, so still aligned with the initial domain focus
- Concerns:
  - only 179 images across 4 scenes
  - likely better as a diversity dataset than a scale dataset
- Initial judgment:
  - excellent architecture testbed for non-pinhole projection handling

### `EnriqueSolarte/mp3d_fpe`

- Why it matters:
  - 360 indoor data with depth information
  - strong test of panoramic geometry support
- Concerns:
  - gated
  - tied to Matterport-derived data and large storage needs
  - may be more useful after basic panoramic support exists
- Initial judgment:
  - strategically valuable, but not an early low-friction pipeline

### `UrbanSyn/UrbanSyn`

- Why it matters:
  - synthetic outdoor driving with depth and sky labels
  - decent scale and likely good geometry
  - useful once the project expands beyond indoor-first priorities
- Concerns:
  - not indoor
  - projection type is still fairly standard compared to panoramic candidates
- Initial judgment:
  - strong outdoor/street candidate after core abstractions stabilize

### `yaraalaa0/TopAir`

- Why it matters:
  - aerial / drone-like nadir viewpoint diversifies the dataset mix
  - synthetic depth and semantic labels should help with `max_dist` treatment for sky
- Concerns:
  - relatively modest scale
  - less central than strong indoor datasets in the near term
- Initial judgment:
  - valuable medium-priority diversification pipeline

### `idsia-robotics/micro-tartanair`

- Why it matters:
  - synthetic ground truth
  - already in WebDataset form
- Concerns:
  - 48x48 resolution makes it poorly aligned with the project’s long-term corpus value
  - better suited to tiny-model benchmarking than to a high-value canonical corpus
- Initial judgment:
  - low priority unless needed as a toy/debugging dataset

## Preliminary Priority Order

### P0

- `GaussianWorld/Hypersim`

### P1

- `theairlabcmu/tartanair`
- `Pointcept/arkitscenes-compressed`
- `Gen3DF/Structured3d-preprocessed`
- `sayakpaul/diode-subset-train`
- `COLE-Ricoh/ToF-360`
- `UrbanSyn/UrbanSyn`
- `yaraalaa0/TopAir`

### P2

- `sayakpaul/nyu_depth_v2`
- `EnriqueSolarte/mp3d_fpe`
- `idsia-robotics/micro-tartanair`

## Why This Order

- `Hypersim` looks like the highest-value indoor scale target if its HF packaging proves usable.
- `tartanair` deserves early attention because it is large, HF-hosted, and recurrent in strong model training recipes.
- `arkitscenes` deserves early attention as a strong real indoor candidate if the compressed HF mirror preserves usable geometry metadata.
- `structured3d` deserves early attention as a large synthetic indoor complement to `Hypersim`.
- `diode-subset-train` looks geometrically attractive, but the current HF packaging appears rough.
- `ToF-360` is disproportionately valuable for projection diversity despite small scale.
- `UrbanSyn` and `TopAir` look valuable once indoor support is solid.
- `nyu_depth_v2` remains useful as a simple legacy/debugging path, but it is de-prioritized because geometry quality matters more than ingestion convenience.
- `micro-tartanair` appears too reduced in resolution to deserve early investment for the main corpus.

## Immediate Research Gaps

- confirm whether there is a better Hugging Face-hosted DIODE package than `sayakpaul/diode-subset-train`
- inspect actual HF file layouts for `Hypersim`, `ToF-360`, and `TopAir`
- identify additional Hugging Face datasets with non-pinhole projection geometry
- decide whether gated Hugging Face datasets should be treated as first-class near-term targets or as secondary work
