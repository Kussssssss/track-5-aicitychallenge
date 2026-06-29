# AI City 2026 Track 5 Pipeline

Format-first tooling for Track 5: build an index from WTS TV2V-style folders,
generate a repeat-last-frame baseline, validate PNG submission structure, and
make a zip whose root contains sample folders directly.

## Expected Test Layout

The official TV2V test package is expected to contain sample folders like:

```text
sample_id/
  input/
    0.png
    1.png
    ...
  caption.json
```

`caption.json` should contain a future frame count, usually under
`"frame length"`, and event captions under `event_phase`.

## Quick Start

Inspect a dataset package:

```bash
python data/inspect_dataset.py --data_root /path/to/track5_test
```

Build a test index:

```bash
python data/build_index.py \
  --data_root /path/to/track5_test \
  --split test \
  --output data/index_test.jsonl
```

Create repeat-last predictions:

```bash
python baselines/repeat_last.py \
  --index data/index_test.jsonl \
  --output outputs/repeat_last
```

Validate:

```bash
python submission/validate_submission.py \
  --index data/index_test.jsonl \
  --pred_dir outputs/repeat_last
```

Zip:

```bash
python submission/make_zip.py \
  --pred_dir outputs/repeat_last \
  --zip_path outputs/submission_repeat_last.zip
```

The zip root will contain:

```text
sample_id/
  0.png
  1.png
  ...
```

not an extra `repeat_last/` wrapper.

