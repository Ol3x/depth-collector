# TODO

## Resume Point

The project can be resumed from repository state alone. A future Codex session does not need the old conversation ID.

Suggested restart prompt:

```text
Continue the depth-collector work from the current repo state.
Current status: docs/spec/contracts are in place, shared runtime skeleton exists, TartanAir scaffold exists but does not yet extract/decode real files.
Next target: make the TartanAir pipeline real in the smallest credible increment.
```

## Next TartanAir Increment

1. Add local ZIP extraction support.
2. Enumerate real extracted files instead of placeholder items.
3. Decode `image_left` frames from extracted files.
4. Generate shared pinhole `ray_dir`.
5. Leave full depth ingestion and real shard writing for a later increment.
