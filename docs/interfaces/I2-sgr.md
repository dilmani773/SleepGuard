# I2: SGR file format (SleepGuard Recording) v1

Owner: Data / ML. Source of truth: `ml/sleepguard_ml/schemas.py` (`Recording`).
Examples: `docs/interfaces/examples/sgr_example_*.json` (synthetic, 7 h each).

- `fs` must be 1. One value per second.
- `spo2`, `hr`, `quality` arrays have the same length. Invalid sample = `null`.
- `source.type`: `sleepguard_clip`, `import_oximeter`, `shhs`, `mesa`, `ucd`, `synthetic`.
- `labels` only for training data.

Changes need a PR approved by everyone who uses the format, and a version bump.
