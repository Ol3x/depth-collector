# State Contract

This document describes the minimum expectations for interruption-tolerance state.

## Required State Areas

The runtime should be able to persist state for:

- download progress
- extraction progress
- processing progress
- error records

## Required Properties

Persistent state should be:

- stage-aware
- append-safe or otherwise corruption-resistant
- stable across restarts
- specific enough to skip already completed work
- specific enough to avoid blind retries of known-bad items

## Typical Identifiers

State entries will likely need identifiers such as:

- archive path or URL
- scene identifier
- shard identifier
- sample identifier
- archive member path

The exact identifier set may vary by dataset, but it should always be explicit and persistent.

## Error Record Linkage

When possible, processing state and error records should be linkable through shared item identifiers so retries and inspections remain straightforward.
