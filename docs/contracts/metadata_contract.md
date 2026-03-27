# Metadata Contract

Each processed dataset should emit `processed/metadata.json`.

## Required Fields

The metadata file should contain enough information to inspect and consume the dataset without scanning every shard.

At minimum it should include:

- dataset name
- pipeline version or format version
- shard count
- shard file names
- number of samples or payload files per shard
- configured `train_val_split`
- suggested training shard list
- suggested validation shard list

## Recommended Fields

The metadata should also aim to include:

- creation timestamp
- source dataset identifier or revision when available
- processing fraction used for the run
- whether the output is a partial-development build or a full build
- summary counts of valid and skipped samples

The metadata contract should stay small but sufficient for debugging and reproducibility.
