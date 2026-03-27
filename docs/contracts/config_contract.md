# Config Contract

The repository should keep one config file per pipeline run.

## Required Fields

The config should eventually define at least:

- dataset identifiers
- processing fraction for development runs
- `train_val_split`
- `max_dist`

One config corresponds to one multi-dataset processing project.

The config should distinguish between:

- project-wide fields
- runtime/output fields
- dataset-specific entries

## `max_dist`

`max_dist` is the maximum representable distance-to-camera value in processed outputs.

It has two roles:

- it caps all finite distances stored in the canonical representation
- it provides the fallback distance for regions that are effectively infinitely far away, such as sky

This means the farthest representable points lie on a sphere of radius `max_dist` around the camera center.

## Multi-Dataset Shape

The config should support enabling multiple datasets in one project run.

Each dataset should have its own config entry, but the canonical geometry rules remain global to the project.
