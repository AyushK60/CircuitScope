"""Task registry: data files (examples/tasks/*.json) describing tasks with
prompt pairs, candidate tokens and recommended hooks. Adding a task needs no
code change. Loading never touches the model; per-pair validation needs the
tokenizer, so it runs lazily on request and is cached."""
