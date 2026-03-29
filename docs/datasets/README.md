# Dataset Notes

This directory contains one short document per dataset integration or candidate dataset.

## Inventory

The table below is the cross-dataset summary to maintain as integrations and research notes evolve.
It should prefer conservative statements over guessed precision; when a field is not yet verified, write
`not yet confirmed`.

| Dataset | Status | Environment | Acquisition Method | Overall Quality | Official Sample Count | Metric | Unit | Minimum Complete Download Unit | License |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [Hypersim](/home/olx2024/repos/depth-collector/docs/datasets/hypersim.md) | implemented | indoor synthetic | artist-authored synthetic renders with full scene/camera metadata | high | about 77.4k images | yes | meters | scene archive; current default config uses `scene_count` | not yet confirmed |
| [TartanAir](/home/olx2024/repos/depth-collector/docs/datasets/tartanair.md) | implemented | mixed synthetic indoor/outdoor navigation scenes | synthetic rendered trajectories | high | not yet confirmed | yes | meters | complete environment+difficulty RGB/depth slice; current default config uses `environment_count` | BSD-3-Clause |
| [TartanGround](/home/olx2024/repos/depth-collector/docs/datasets/tartanground.md) | implemented | mixed synthetic ground robotics scenes | synthetic rendered trajectories | high | not yet confirmed | yes | meters | complete environment/version/trajectory/camera RGB/depth slice; current default config uses `environment_count` | not yet confirmed |
| [MegaDepth](/home/olx2024/repos/depth-collector/docs/datasets/megadepth.md) | candidate | mostly outdoor Internet landmarks | Internet photos + SfM/MVS reconstruction | medium to high | 196 reconstructed locations (official) | no | not applicable | not yet confirmed; likely one scene if per-scene HF files are exposed, otherwise bundle-scale | CC BY 4.0 for depth/SfM outputs; original images retain their own licenses |
| [Structured3D](/home/olx2024/repos/depth-collector/docs/datasets/structured3d.md) | candidate | indoor synthetic | synthetic interior renders | high | not yet confirmed | yes | meters | effectively full reconstructed archive, about 307 GB; 308 ~1 GB chunks are partial fragments, not complete units | restrictive / upstream Structured3D terms |
| [Virtual KITTI 2](/home/olx2024/repos/depth-collector/docs/datasets/virtual_kitti_2.md) | candidate | outdoor synthetic driving | synthetic rendered driving scenes | high | not yet confirmed | yes | meters | not yet confirmed; no suitable verified HF raw package pinned down | not yet confirmed |
| [UrbanSyn](/home/olx2024/repos/depth-collector/docs/datasets/urbansyn.md) | candidate | outdoor synthetic urban | synthetic rendered urban scenes | high | not yet confirmed | yes | not yet confirmed | one aligned RGB+depth frame pair, about 11.1 MB total on reviewed HF tree | CC-BY-SA-4.0 |
| [ARKitScenes](/home/olx2024/repos/depth-collector/docs/datasets/arkitscenes.md) | candidate | indoor real | RGB + depth / LiDAR capture | medium | not yet confirmed | yes | meters | one `arkitscenes_*.tar.gz` shard, about 9.5 GB on reviewed HF tree | not clearly stated on reviewed HF page |
| [DIODE subset train](/home/olx2024/repos/depth-collector/docs/datasets/diode_subset_train.md) | candidate | indoor and outdoor real | active sensor depth capture | medium to high | not yet confirmed | yes | meters | one `train_subset.tar.gz`, about 12.8 GB | MIT |
| [WMGStereo](/home/olx2024/repos/depth-collector/docs/datasets/wmg_stereo.md) | implemented | synthetic stereo across flying / nature / indoor categories | stereo RGB + disparity + camera calibration | medium to high | not yet confirmed | no | relative `[0, 1]` | category-specific `.tar.gz` archives; exact smallest complete unit not yet confirmed | not yet confirmed |
| [NYU Depth V2](/home/olx2024/repos/depth-collector/docs/datasets/nyu_depth_v2.md) | candidate | indoor real | Kinect-style RGB-D capture | low to medium | about 1.4k labeled pairs | yes | meters | smallest visible shard is `val-000001.tar` at about 14.8 MB; needs implementation-time verification that it is self-contained enough for a complete smoke unit | Apache-2.0 for reviewed HF package |
| [Taskonomy](/home/olx2024/repos/depth-collector/docs/datasets/taskonomy.md) | candidate | indoor synthetic | synthetic indoor renders | not yet confirmed | not yet confirmed | not yet confirmed | not yet confirmed | not yet confirmed; no suitable verified HF raw package pinned down | not yet confirmed |
| [ToF-360](/home/olx2024/repos/depth-collector/docs/datasets/tof_360.md) | candidate | indoor real 360 | time-of-flight capture | not yet confirmed | not yet confirmed | yes | not yet confirmed | one scene folder appears to be the natural complete unit; reviewed sources confirm only 4 scenes total, but byte size was not verified in this pass | CC-BY-NC-SA-4.0 |
| [TopAir](/home/olx2024/repos/depth-collector/docs/datasets/topair.md) | candidate | outdoor aerial / remote sensing | aerial capture / derived geometry | not yet confirmed | not yet confirmed | not yet confirmed | not yet confirmed | likely one trajectory folder such as `AccuCities_1`; full dataset is about 4.02 GB, so per-trajectory size is likely well below 1 GB but not yet confirmed | not yet confirmed |
| [MP3D-FPE](/home/olx2024/repos/depth-collector/docs/datasets/mp3d_fpe.md) | candidate | indoor real | Matterport-style scan / reconstruction derivative | not yet confirmed | not yet confirmed | not yet confirmed | not yet confirmed | not yet confirmed from gated listing; full dataset storage note is about 320 GB | MIT with upstream terms |
| [Micro-TartanAir](/home/olx2024/repos/depth-collector/docs/datasets/micro_tartanair.md) | candidate | mixed synthetic indoor/outdoor navigation scenes | synthetic rendered trajectories | medium | subset / not yet confirmed | yes | meters | one tarball; smallest reviewed complete unit is `tartanair_hard_48x48.tar.gz` at about 1.65 GB | CC-BY-4.0 via TartanAir derivation |

## Per-Dataset Notes

Each dataset note should summarize:

- what the dataset contains
- why it matters
- what is known about its geometry conventions
- likely integration difficulty
- current status and issues

These notes should support both implementation work and roadmap decisions.

## Candidate Priority Notes

When prioritizing untackled candidate datasets, optimize jointly for:

- high geometry quality
- high quantity / coverage
- small complete non-partial download units

That means tiny shards alone are not enough. A dataset with weak geometry should rank low even if it is easy to
download, while a strategically strong dataset can still rank well if its acquisition unit is somewhat larger.

Current rough priority by the combined rule above:

1. UrbanSyn
2. Structured3D
3. ARKitScenes
4. DIODE subset train
5. TopAir
6. ToF-360
7. Micro-TartanAir
8. MP3D-FPE
9. NYU Depth V2

Unranked for now:

- `Taskonomy`: strategically important, but no suitable verified HF raw package is pinned down yet.
- `Virtual KITTI 2`: strategically important for outdoor synthetic driving, but no suitable verified HF raw package is
  pinned down yet.

Notes:

- `UrbanSyn` currently looks like the best balance of quality, scale potential, and tiny complete acquisition units.
- `Structured3D` stays near the top because of likely very high quality and scale, even though its current HF mirror is
  painful for non-partial acquisition; this is a strategic ranking, not a pure smoke-test ranking.
- `ARKitScenes` and `DIODE subset train` rank above smaller convenience datasets because real-data quality and corpus
  value matter more than raw download speed alone.
- `TopAir` and `ToF-360` remain promising because they appear to offer useful diversity with manageable acquisition
  units, but their exact smallest complete sizes still need implementation-time verification.
- `Micro-TartanAir` is useful for debugging and fast end-to-end tests, but its reduced resolution lowers its strategic
  value for the main corpus.
- `MP3D-FPE` stays low because it is gated, large, and more complex, despite being potentially valuable.
- `NYU Depth V2` should remain last because the geometry quality is poor for this project; flying-pixel artifacts make
  it a weak target even though the visible HF shard size is attractive.
