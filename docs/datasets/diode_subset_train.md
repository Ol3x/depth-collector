# DIODE Subset Train

- Dataset: `sayakpaul/diode-subset-train`
- Reviewed: 2026-03-27
- Hugging Face: https://huggingface.co/datasets/sayakpaul/diode-subset-train
- Domain: indoor + outdoor, real
- Projection: likely standard perspective RGB-D
- Scale signal: medium subset
- Geometry assessment: promising, based on DIODE’s stated focus on accurate dense long-range depth
- Artifact risk: unknown from HF packaging alone
- Canonical conversion difficulty: medium
- License on HF card: MIT
- Status: candidate
- Priority tier: P1
- Why it matters: potentially strong real-data geometry with cross-domain value
- Known issues: subset packaging only; HF viewer/split parsing appears broken
- HF packaging notes: current repo appears to expose a single `train_subset.tar.gz`
- Pipeline notes: validity masks are a strong fit for this project’s conservative invalid-data policy if the HF package preserves them
