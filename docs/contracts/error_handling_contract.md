# Error Handling Contract

The processed dataset should favor clean outputs over ambiguous partial repairs.

## Policy

- do not emit validity masks in the canonical sample format for now
- repair invalid data only when the repair has a clear semantic basis
- if a sample cannot be repaired meaningfully, do not include it in processed outputs
- record failures in persistent error artifacts for later inspection
- maintain error tracking for download, extraction, and processing stages

## Examples

- if a dataset provides sky masks and depth is infinite or undefined in those regions, mapping those pixels to `max_dist` is acceptable
- if a sample has corrupted geometry with no principled recovery path, the sample should be rejected and logged

Judgment is required here, but it should be conservative.

## Minimum Error Record Content

Each error record should contain at least:

- pipeline stage
- dataset identifier
- file or item identifier that caused the failure
- error message

Optional fields may include timestamps, traceback details, retry counts, and status flags.
