import argparse
import json
from pathlib import Path

from PIL import Image


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="Repeat last history frame as Track 5 baseline.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    count = 0
    for sample in load_jsonl(args.index):
        sample_id = sample["sample_id"]
        future_len = int(sample["future_len"])
        width, height = int(sample["width"]), int(sample["height"])
        history = sample.get("history_frame_paths") or []
        if not history:
            raise ValueError(f"{sample_id}: missing history_frame_paths")

        case_dir = output_root / sample_id
        case_dir.mkdir(parents=True, exist_ok=True)
        existing = list(case_dir.glob("*.png"))
        if existing and not args.overwrite:
            raise FileExistsError(f"{case_dir} already contains PNG files. Use --overwrite.")

        with Image.open(history[-1]) as img:
            img = img.convert("RGB")
            if img.size != (width, height):
                img = img.resize((width, height), Image.Resampling.BICUBIC)
            for i in range(future_len):
                img.save(case_dir / f"{i}.png")
        count += 1

    print(f"generated repeat-last predictions for {count} samples at {output_root}")


if __name__ == "__main__":
    main()

