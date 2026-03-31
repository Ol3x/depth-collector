# MegaDepth

- Dataset: `MegaDepth`
- Reviewed: 2026-03-27
- Official page: https://www.cs.cornell.edu/projects/megadepth/
- Paper: https://openaccess.thecvf.com/content_cvpr_2018/html/Li_MegaDepth_Learning_Single-View_CVPR_2018_paper.html
- Domain: mostly outdoor Internet landmark photography
- Projection: perspective Internet photos with reconstructed camera intrinsics/extrinsics
- Scale signal: high
- Geometry assessment: strategically valuable, but noisier than clean synthetic sources because supervision is derived from SfM/MVS rather than direct sensing or renderer ground truth
- Artifact risk: medium to high; the paper itself discusses noise, unreconstructable objects, and the need for cleaning
- Canonical conversion difficulty: medium to high
- License: CC BY 4.0 for MegaDepth depth maps and SfM models; original images retain their own licenses
- Status: candidate
- Priority tier: near-term candidate
- Why it matters: very common monocular-depth training source, large scale, and a good stress test for real-world Internet-photo geometry rather than sensor-clean or renderer-clean supervision
- Known issues:
  - official sample count in images is not yet pinned down from a primary source in this repo; the official page clearly states 196 reconstructed locations
  - geometry comes from COLMAP SfM/MVS, so noisy regions and incomplete reconstructions are expected
  - metric status is unlikely to hold globally because Internet-photo SfM/MVS has no absolute scale anchor
- Geometry notes:
  - the dataset is generated from multi-view Internet photo collections using structure-from-motion and multi-view stereo
  - camera intrinsics and extrinsics are available through the released SfM models
  - metric scale should be treated as not guaranteed; this is an inference from the acquisition method and should be validated against the actual released metadata before implementation
  - under the current project rule, non-metric MegaDepth exports should normalize radial distance into `[0, 1]`, with `1` used as the far / max bucket

## Current Pipeline Direction

- The repository includes a MegaDepth pipeline and config contract, but this dataset should still be treated as a candidate until the full live acquisition path is exercised more broadly.
- The current pipeline model uses bundle downloads plus scene selection from scene-info metadata.

## Minimum Readable Selection

- `selection: "minimum_readable"` means the smallest selected MegaDepth candidate set that still yields at least one readable sample with:
  - source RGB
  - usable depth or reconstruction-derived geometry
  - camera metadata sufficient to derive canonical `ray_dir`
- In the current pipeline shape, that usually means one scene from the ordered scene pool and one bundle from the ordered bundle pool.
- `selection: "all"` means all selected scenes and bundles.
- A ratio in `(0, 1]` means the corresponding prefix of the ordered candidate pools.
