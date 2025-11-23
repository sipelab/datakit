# Sample Experiment Fixtures

The `sample_experiment*` folders exist as lightweight fixtures for validating discovery and inventory logic. Each folder mirrors a different layout nuance we expect to encounter in the field.

- `sample_experiment1` – Minimal single-session layout. One subject (`STREHAB07`, session `05`) with processed mesoscope mean traces, DeepLabCut output, and raw behavior files (`dataqueue`, treadmill), metadata JSON, configuration CSV, and notes.
- `sample_experiment2` – Structure identical to `sample_experiment1` yet missing timeline.csv and dataqueue.csv files to represent Mesofield output before version 0.9.0
- `sample_experiment3` – Multi-record session for subject `ACUTEVIS06`. Contains two dataqueue, wheel, and Psychopy files in the same session directory, plus suite2p output. Exercises manifest grouping when multiple files share the same tag within a session.

Each manifest stores paths relative to the experiment root. The discovery helpers normalise Windows-style separators during inventory building so these fixtures remain portable across platforms.
