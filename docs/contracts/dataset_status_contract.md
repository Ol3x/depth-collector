# Dataset Status Contract

Each dataset integration should eventually have a short status document under `docs/datasets/`.
The `docs/datasets/README.md` index should also maintain a cross-dataset table with the key comparison fields.

## Purpose

The status document should make it easy to understand:

- what the dataset is
- where it comes from
- how mature the integration is
- what issues or risks are known
- whether the dataset is worth continued investment

## Suggested Fields

- dataset name
- source location
- Hugging Face location
- environment or domain
- acquisition method
- modality summary
- geometry summary
- geometric quality assessment
- official dataset sample count, or `not yet confirmed`
- whether the dataset is metric
- metric unit, when applicable
- notable artifact risks such as flying pixels
- expected canonical conversion difficulty
- license
- current integration status
- known issues
- maintenance notes
- priority or roadmap tier

The document should stay concise but decision-useful.

New dataset notes should ideally be created from the standard candidate template and informed by the triage checklist so that comparisons remain consistent.
