import argparse
import json
from pathlib import Path

from PIL import Image


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc


def validate(index_path, pred_dir, strict_extras=False):
    pred_dir = Path(pred_dir)
    if not pred_dir.is_dir():
        raise NotADirectoryError(pred_dir)

    samples = list(load_jsonl(index_path))
    expected_ids = {s["sample_id"] for s in samples}
    actual_dirs = {p.name for p in pred_dir.iterdir() if p.is_dir()}

    missing_dirs = sorted(expected_ids - actual_dirs)
    extra_dirs = sorted(actual_dirs - expected_ids)
    errors = []
    if missing_dirs:
        errors.append(f"Missing sample folders: {missing_dirs[:10]}{' ...' if len(missing_dirs) > 10 else ''}")
    if strict_extras and extra_dirs:
        errors.append(f"Unexpected sample folders: {extra_dirs[:10]}{' ...' if len(extra_dirs) > 10 else ''}")

    checked_frames = 0
    for sample in samples:
        sample_id = sample["sample_id"]
        case_dir = pred_dir / sample_id
        if not case_dir.is_dir():
            continue

        expected_size = (int(sample["width"]), int(sample["height"]))
        future_len = int(sample["future_len"])
        expected_names = {f"{i}.png" for i in range(future_len)}
        actual_pngs = {p.name for p in case_dir.glob("*.png")}

        missing_frames = sorted(expected_names - actual_pngs, key=lambda x: int(Path(x).stem))
        if missing_frames:
            errors.append(
                f"{sample_id}: missing frames {missing_frames[:10]}"
                f"{' ...' if len(missing_frames) > 10 else ''}"
            )
        if strict_extras:
            extra_frames = sorted(actual_pngs - expected_names)
            if extra_frames:
                errors.append(
                    f"{sample_id}: unexpected PNG frames {extra_frames[:10]}"
                    f"{' ...' if len(extra_frames) > 10 else ''}"
                )

        for i in range(future_len):
            frame_path = case_dir / f"{i}.png"
            if not frame_path.exists():
                continue
            try:
                with Image.open(frame_path) as img:
                    if img.size != expected_size:
                        errors.append(f"{frame_path}: got size {img.size}, expected {expected_size}")
                    if img.format != "PNG":
                        errors.append(f"{frame_path}: image format is {img.format}, expected PNG")
            except Exception as exc:
                errors.append(f"{frame_path}: cannot read image: {exc}")
            checked_frames += 1

    if errors:
        print(f"Invalid submission: {len(errors)} error(s)")
        for error in errors[:100]:
            print(f"- {error}")
        if len(errors) > 100:
            print(f"... {len(errors) - 100} more error(s)")
        raise SystemExit(1)

    print(f"Valid submission format. samples={len(samples)} frames={checked_frames}")


def main():
    parser = argparse.ArgumentParser(description="Validate AI City Track 5 PNG submission folders.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--strict_extras", action="store_true")
    args = parser.parse_args()
    validate(args.index, args.pred_dir, strict_extras=args.strict_extras)


if __name__ == "__main__":
    main()

