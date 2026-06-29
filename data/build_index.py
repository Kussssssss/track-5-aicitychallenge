import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prompt_builder import build_prompt, normalize_text, phase_label_from_id


FUTURE_DIR_NAMES = ("future", "target", "targets", "output", "outputs", "gt")
FUTURE_LEN_KEYS = ("frame length", "frame_length", "future_len", "num_future_frames", "N", "n")


def numeric_sort_key(path):
    stem = Path(path).stem
    if stem.isdigit():
        return (0, int(stem), stem)
    numbers = re.findall(r"\d+", stem)
    if numbers:
        return (1, int(numbers[-1]), stem)
    return (2, stem)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_present(mapping, keys):
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def parse_future_len(caption, default_future_len=None):
    value = first_present(caption, FUTURE_LEN_KEYS)
    if value is None:
        value = first_present(caption.get("metadata", {}), FUTURE_LEN_KEYS)
    if value is None:
        if default_future_len is None:
            return None
        return int(default_future_len)
    return int(value)


def parse_event_phase(caption):
    phases = caption.get("event_phase") or caption.get("event_phases") or []
    if isinstance(phases, dict):
        phases = [phases]
    if not isinstance(phases, list):
        phases = []

    phase = phases[0] if phases else caption
    labels = phase.get("labels") or phase.get("label") or phase.get("phase") or []
    if isinstance(labels, list):
        phase_id = labels[0] if labels else None
    else:
        phase_id = labels

    caption_pedestrian = (
        phase.get("caption_pedestrian")
        or phase.get("pedestrian_caption")
        or phase.get("pedestrian")
        or caption.get("caption_pedestrian")
        or caption.get("pedestrian_caption")
        or ""
    )
    caption_vehicle = (
        phase.get("caption_vehicle")
        or phase.get("vehicle_caption")
        or phase.get("vehicle")
        or caption.get("caption_vehicle")
        or caption.get("vehicle_caption")
        or ""
    )

    return phase_id, normalize_text(caption_pedestrian), normalize_text(caption_vehicle)


def find_caption(input_dir):
    """Locate caption.json for a given input/ dir, tolerating layout variants:
    caption inside input/, or beside input/ (in the sample root)."""
    for cand in (input_dir / "caption.json", input_dir.parent / "caption.json"):
        if cand.is_file():
            return cand
    return None


def discover_samples(root):
    """Input-dir-centric discovery. Returns (sample_root, input_dir, caption_path).

    `sample_root` (parent of input/) names the prediction folder; this is robust
    whether caption.json sits inside input/ or beside it."""
    samples = []
    for input_dir in sorted(root.rglob("input")):
        if not input_dir.is_dir() or not any(input_dir.glob("*.png")):
            continue
        caption_path = find_caption(input_dir)
        samples.append((input_dir.parent, input_dir, caption_path))
    return samples


def find_future_frames(sample_dir):
    for name in FUTURE_DIR_NAMES:
        future_dir = sample_dir / name
        if future_dir.is_dir():
            frames = sorted(future_dir.glob("*.png"), key=numeric_sort_key)
            if frames:
                return frames
    return []


def build_rows(data_root, split, default_future_len=None, sample_id_mode="name"):
    data_root = Path(data_root).resolve()
    samples = discover_samples(data_root)
    if not samples:
        raise RuntimeError(
            f"No sample folders found under {data_root}. Expected input/*.png (+ caption.json)."
        )

    rows = []
    seen_ids = set()
    missing_caption = 0
    for sample_dir, input_dir, caption_path in samples:
        if caption_path is None:
            missing_caption += 1
            if default_future_len is None:
                raise ValueError(
                    f"No caption.json found for {input_dir}. Pass --default_future_len "
                    f"to index caption-less samples, or check the dataset layout."
                )
            caption = {}
        else:
            caption = load_json(caption_path)
        history_frames = sorted(input_dir.glob("*.png"), key=numeric_sort_key)
        future_frames = find_future_frames(sample_dir)

        if sample_id_mode == "relative":
            sample_id = str(sample_dir.relative_to(data_root)).replace("\\", "__").replace("/", "__")
        else:
            sample_id = sample_dir.name
        if sample_id in seen_ids:
            sample_id = str(sample_dir.relative_to(data_root)).replace("\\", "__").replace("/", "__")
        seen_ids.add(sample_id)

        future_len = parse_future_len(caption, default_future_len=default_future_len)
        if future_len is None:
            if future_frames:
                future_len = len(future_frames)
            else:
                raise ValueError(
                    f"Missing future length in {caption_path}. Pass --default_future_len if needed."
                )

        with Image.open(history_frames[0]) as img:
            width, height = img.size

        phase_id, caption_pedestrian, caption_vehicle = parse_event_phase(caption)
        phase_label = phase_label_from_id(phase_id)
        prompt = build_prompt(caption_pedestrian, caption_vehicle, phase_label)

        row = {
            "sample_id": sample_id,
            "split": split,
            "sample_dir": str(sample_dir),
            "caption_path": None if caption_path is None else str(caption_path),
            "history_frame_paths": [str(p) for p in history_frames],
            "future_frame_paths": [str(p) for p in future_frames],
            "future_len": int(future_len),
            "width": int(width),
            "height": int(height),
            "phase_id": None if phase_id is None else str(phase_id),
            "phase_label": phase_label,
            "caption_pedestrian": caption_pedestrian,
            "caption_vehicle": caption_vehicle,
            "prompt": prompt,
        }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build JSONL index for AI City Track 5 TV2V samples.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--default_future_len", type=int, default=None)
    parser.add_argument(
        "--sample_id_mode",
        choices=["name", "relative"],
        default="name",
        help="Use folder name unless duplicates exist; relative always uses root-relative path.",
    )
    args = parser.parse_args()

    rows = build_rows(
        args.data_root,
        args.split,
        default_future_len=args.default_future_len,
        sample_id_mode=args.sample_id_mode,
    )
    write_jsonl(rows, Path(args.output))
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

