import argparse
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create Track 5 submission zip without extra root nesting.")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--zip_path", required=True)
    parser.add_argument("--compression", choices=["stored", "deflated"], default="deflated")
    parser.add_argument(
        "--top_folder",
        default="",
        help="Optional top-level folder inside the zip (e.g. 'prediction'). "
        "Default: sample folders at the zip root.",
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir).resolve()
    zip_path = Path(args.zip_path).resolve()
    if not pred_dir.is_dir():
        raise NotADirectoryError(pred_dir)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    compression = zipfile.ZIP_STORED if args.compression == "stored" else zipfile.ZIP_DEFLATED
    files = sorted(p for p in pred_dir.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError(f"No files found under {pred_dir}")

    top = args.top_folder.strip("/")
    with zipfile.ZipFile(zip_path, "w", compression=compression) as zf:
        for path in files:
            rel = path.relative_to(pred_dir).as_posix()
            arcname = f"{top}/{rel}" if top else rel
            zf.write(path, arcname)

    print(f"wrote {zip_path} with {len(files)} files")


if __name__ == "__main__":
    main()

