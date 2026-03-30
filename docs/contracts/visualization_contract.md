# Visualization Contract

This document defines the repository-wide rule for processed-sample visualization.

## Main Rule

The repository has one official visualization path for processed datasets.

That path is the shared visualization module invoked by `dc visualize`.

## Shared Responsibilities

The shared visualization layer should own:

- loading processed samples from shards
- reconstructing geometry from `distance * ray_dir`
- rendering the standard diagnostic panels
- grouping samples into output folders
- writing visualization images under the standard dataset-local `visualizations/` directory

## Dataset Pipeline Limit

Concrete dataset pipelines are not allowed to:

- define their own visualization command or hook
- render their own diagnostic image sheets
- invent a dataset-local panel layout or color-mapping convention
- bypass the shared visualization implementation for normal repository usage

Dataset pipelines may only contribute standard processed samples and provenance fields that the shared visualization module already knows how to consume.

## Evolution Rule

If visualization needs to change, it should change centrally in the shared visualization module so that all datasets continue to use the same method.

New datasets do not get to introduce alternate visualization behavior by default.
