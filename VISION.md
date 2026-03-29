The goal of `depth-collector` is to build a durable pipeline system that converts multiple public datasets into one large, accessible, high-quality, standardized monocular distance corpus.

The project is not only a collection of one-off scripts. It should become a reusable framework for:

- downloading source datasets from Hugging Face Hub or other public mirrors
- normalizing heterogeneous camera conventions into one shared geometric standard
- producing sharded training data in a format that is practical for large-scale model training
- documenting every assumption well enough that future agents can extend the system safely

The project should also be opinionated about responsibility boundaries:

- users should not be expected to manually gather source datasets for a completed integration
- if a dataset is considered supported, the repository should know how to fetch it from a public source itself
- local manual staging is acceptable only as an exploratory or temporary development step, not as the finished product

The ambition is to support many datasets over time while preserving one strict sample contract:

- RGB image
- distance-to-camera target
- ray direction field

That contract should remain honest about scale:

- metric datasets should preserve metric camera distance
- non-metric datasets should still use camera-distance semantics, but normalized into `[0, 1]`
- for non-metric datasets, `1` is the far / max bucket rather than a physical unit

The main product is therefore a standardized dataset factory, not a single dataset export.

The project should also be treated as a live system operating in a changing ecosystem:

- some datasets will appear over time
- some datasets or revisions may disappear from Hugging Face Hub or other sources
- temporary loss of support for a disappearing dataset is acceptable
- long-term archival completeness is not the current objective
- fast integration of new useful datasets is a priority

The project should therefore optimize for adaptability and extension speed, not only for permanence.

Because the project is implemented primarily by supervised LLM agents, the codebase and documentation should optimize for:

- explicit invariants over implicit conventions
- small, composable abstractions
- precise file and data contracts
- clear extension points for adding a new dataset pipeline
- low ambiguity in terminology, especially around geometry
