# Dataset Triage Checklist

This checklist should be used before promoting a new dataset candidate into the active roadmap.

The goal is consistency. A dataset should not move into `P0` or `P1` just because it sounds promising.

## Gate 1: Scope Fit

Check:

- is the dataset available on Hugging Face
- is it plausibly useful for monocular depth or monocular geometry learning
- is it consistent with the current project scope

If any answer is clearly no, the dataset should not enter the active roadmap.

## Gate 2: Geometry Quality

Check:

- does the dataset appear to have strong geometric supervision
- is the geometry definition understandable
- is there evidence of severe flying-point artifacts or unreliable 3D
- are invalid regions documented clearly enough to support conservative handling

This is the primary gate. Weak geometry should block prioritization even if the dataset is large.

## Gate 3: Canonical Conversion Feasibility

Check:

- can we derive or reconstruct a camera model
- can we generate unit `ray_dir`
- can we derive canonical `distance`
- is the source definition radial distance, z-depth, disparity, or something more ambiguous
- do we have enough metadata to make the conversion principled

If the conversion path is not credible, the dataset should remain a research target rather than an implementation target.

## Gate 4: Hugging Face Packaging Quality

Check:

- is there a concrete HF-hosted package
- is the file layout inspectable
- is the package raw enough to preserve useful geometry metadata
- is the package operationally reasonable to download and process

Good dataset, bad HF packaging is still a real problem for this project.

## Gate 5: Operational Fit

Check:

- can the dataset be processed incrementally
- is there a meaningful processing-item boundary
- can interruption tolerance be implemented cleanly
- is the package size manageable for development-time partial runs

Datasets that force brittle all-or-nothing workflows should be deprioritized.

## Gate 6: Strategic Value

Check:

- does the dataset add strong scale
- does it improve projection-model diversity
- does it improve domain diversity
- does it recur in strong modern model training recipes
- does it complement the current selected mix

This stage matters only after the geometry and feasibility gates are satisfied.

## Output Decision

After the checklist, assign one of:

- `reject`: not a useful target for the current project
- `watchlist`: strategically interesting, but blocked by quality, packaging, or metadata uncertainty
- `P2`: plausible but lower-value or more speculative target
- `P1`: strong candidate after core abstractions are stable
- `P0`: near-term top priority

## Minimum Evidence To Record

Every triage pass should record:

- the reviewed HF package or packages
- the main geometry signal
- the main packaging signal
- the main risk
- the resulting tier decision
- the review date
