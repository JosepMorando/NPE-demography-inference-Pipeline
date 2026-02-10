# CHANGELOG v8.4

## Scope

This file replaces legacy explanatory changelogs from earlier versions and
summarizes the current maintained state of the pipeline.

## Current pipeline status

- Active pipeline directory is `final_v8.4/clean_pipeline_v8.4`.
- Documentation now points to v8.4 paths and filenames.
- Legacy versioned changelog narratives were removed to reduce confusion.

## Operational notes

- Use `scripts/run_pod_test.sh` before production runs.
- Use `scripts/run_production.sh` for single-node production.
- Use `scripts/run_production_multinode*.sh` for distributed simulation runs.

## Breaking documentation changes

- `README_v8.1.md` replaced by `README.md` in this directory.
- Old changelog files (`CHANGELOG.md`, `CHANGELOG_v6.1.md`,
  `CHANGELOG_v7.md`, `CHANGELOG_v8.md`, `CHANGELOG_v8.1.md`) were removed.
