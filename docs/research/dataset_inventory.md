# Dataset Inventory

This document should become the working inventory of candidate source datasets.

## Purpose

The project needs a maintained view of the available monocular depth-style dataset landscape in order to:

- decide which pipelines to build next
- estimate integration complexity
- reason about coverage gaps
- react quickly when new datasets appear

## Per-Dataset Information To Capture

For each candidate dataset, the inventory should aim to track:

- name
- source or hosting location
- Hugging Face location
- task/domain type
- approximate scale
- sensor or rendering origin
- camera model information quality
- presence or absence of sky or invalid-value annotations
- expected difficulty of converting to canonical `distance` plus unit `ray_dir`
- expected data quality
- expected strategic value for the unified corpus
- license
- current availability status

## Evaluation Dimensions

Candidate datasets should be judged along at least these axes:

- geometric reliability
- absence or severity of artifacts such as flying pixels
- visual diversity
- scale
- annotation completeness
- Hugging Face accessibility and maintenance practicality
- integration complexity
- likely downstream value

## Quality Interpretation

In this project, dataset quality refers primarily to geometric quality.

- strong 3D geometry is more important than image aesthetics
- visually mediocre images may still be valuable if their geometry is reliable
- severe geometric artifacts, especially flying pixels, are a major negative signal

## Source Scope

For the current phase, candidate datasets should be sourced exclusively from Hugging Face.

This inventory may mention external origins for context, but active pipeline targeting should focus on datasets that are available through Hugging Face.

This document is expected to evolve continuously.

New candidate additions should follow the standard research workflow and triage checklist before they materially affect prioritization.
