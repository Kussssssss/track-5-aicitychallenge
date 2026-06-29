import argparse
import json
from collections import Counter
from pathlib import Path


def count_files(root, suffix):
    return sum(1 for _ in root.rglob(f"*{suffix}"))


def main():
    parser = argparse.ArgumentParser(description="Inspect a Track 5 dataset folder.")
    parser.add_argument("--data_root", required=True, help="Dataset root to inspect.")
    parser.add_argument("--max_items", type=int, default=50)
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    suffix_counts = Counter(p.suffix.lower() for p in root.rglob("*") if p.is_file())
    caption_files = sorted(root.rglob("caption.json"))

    # input-dir-centric discovery (matches data/build_index.py)
    input_dirs = [d for d in sorted(root.rglob("input")) if d.is_dir() and any(d.glob("*.png"))]
    cap_inside = sum(1 for d in input_dirs if (d / "caption.json").is_file())
    cap_beside = sum(1 for d in input_dirs if (d.parent / "caption.json").is_file())
    sample_dirs = [d.parent for d in input_dirs]

    print(f"root: {root}")
    print(f"input/ dirs with PNGs: {len(input_dirs)}")
    print(f"  caption.json INSIDE input/: {cap_inside}")
    print(f"  caption.json BESIDE input/ (sample root): {cap_beside}")
    print(f"json files: {count_files(root, '.json')}")
    print(f"mp4 files: {count_files(root, '.mp4')}")
    print(f"png files: {count_files(root, '.png')}")
    print("top suffix counts:")
    for suffix, count in suffix_counts.most_common(20):
        print(f"  {suffix or '<none>'}: {count}")

    print("\nsample directories:")
    for sample_dir in sample_dirs[: args.max_items]:
        rel = sample_dir.relative_to(root)
        inputs = sorted((sample_dir / "input").glob("*.png"))
        future_dirs = [
            name
            for name in ("future", "target", "targets", "output", "outputs", "gt")
            if (sample_dir / name).is_dir()
        ]
        print(f"  {rel} | input_png={len(inputs)} | future_dirs={future_dirs}")

    if caption_files:
        first = caption_files[0]
        print(f"\nfirst caption.json: {first.relative_to(root)}")
        try:
            data = json.loads(first.read_text(encoding="utf-8"))
            keys = ", ".join(sorted(data.keys()))
            print(f"caption keys: {keys}")
            print(json.dumps(data, ensure_ascii=False, indent=2)[:2000])
        except Exception as exc:
            print(f"failed to parse caption: {exc}")


if __name__ == "__main__":
    main()

