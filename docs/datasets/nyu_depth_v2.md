# NYU Depth V2

- Dataset: `sayakpaul/nyu_depth_v2`
- Reviewed: 2026-03-27
- Hugging Face: https://huggingface.co/datasets/sayakpaul/nyu_depth_v2
- Domain: indoor, real RGB-D
- Projection: likely standard perspective RGB-D capture
- Scale signal: small
- Geometry assessment: legacy indoor RGB-D source with weak geometric quality for this project
- Artifact risk: high concern for flying 3D points and unreliable filled-depth regions
- Canonical conversion difficulty: low to medium
- License on HF card: Apache-2.0 for the HF package, with card text also referring to original/preprocessed dataset licensing
- Status: candidate
- Priority tier: P2
- Why it matters: still useful as a low-friction legacy/debugging indoor dataset
- Known issues: small scale; not likely to be a long-term anchor dataset by itself; geometry quality is not strong enough for a top-priority role here
- HF packaging notes: custom HF dataset script over `.tar` shards containing `.h5` payloads
- Pipeline notes: likely straightforward streaming pipeline, but the project should treat it as a convenience/debugging target rather than a high-value geometry target
