# Interruption Tolerance

The processing system should be designed to survive restarts and partial failures.

## Goals

- avoid repeating completed downloads
- avoid repeating completed extraction work
- avoid reprocessing source items that already produced valid outputs
- avoid retrying known-bad items without a reason

## Expected Mechanisms

The implementation will likely need persistent state such as:

- download manifests
- extraction manifests
- processed-item registries
- shard write checkpoints
- error registries

These artifacts may live per dataset under `data/<project_name>/<dataset_name>/state/`.

Error registries should be stage-aware and should record, at minimum, the failing file or item and the error message.

## Design Intent

Interruption tolerance is not an optional optimization. It is part of the operating model because the project is expected to handle large datasets and long-running jobs.
