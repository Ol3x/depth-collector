# Live Project Philosophy

`depth-collector` should be treated as a live project that tracks a changing public dataset ecosystem.

## What This Means

- public datasets may appear, change, move, or disappear
- support for a dataset is useful, but not guaranteed forever
- the project does not currently aim to maintain a permanent archive of every upstream source
- temporary loss of functionality caused by upstream disappearance is acceptable

## Strategic Priority

The repository should prioritize:

- rapid integration of new datasets
- clear extension patterns
- low-friction maintenance when upstream sources change

It should not over-optimize for archival permanence at the cost of adaptability.

## Architectural Consequence

This philosophy strengthens the case for:

- clean dataset-specific modules
- explicit per-dataset documentation
- shared lifecycle abstractions
- minimal coupling between dataset integrations

The easier it is to add or remove one dataset pipeline without disturbing others, the healthier the project will be.
