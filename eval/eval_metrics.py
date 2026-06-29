"""Offline evaluation for AI City 2026 Track 5 (TV2V future prediction).

Computes the five challenge metrics -- PSNR, SSIM, LPIPS, CLIP-S, FVD -- between
predicted frames and ground-truth future frames, using a JSONL index built by
data/build_index.py on the *val* split (where GT `future_frame_paths` exist).

The official evaluation script is not public, so these are standard, widely used
implementations; the exact CLIP-S text and FVD backbone are documented below and
configurable so they can be matched to the organizers' definition if published.

  PSNR  (higher better)  pixel reconstruction, per-frame, data_range=255
  SSIM  (higher better)  structural similarity, per-frame (skimage)
  LPIPS (lower  better)  perceptual distance (lpips, AlexNet by default)
  CLIP-S(higher better)  CLIP image-text cosine x100, generated frame vs caption
  FVD   (lower  better)  Frechet Video Distance over the whole set (cd-fvd, I3D)

Usage:
    python eval/eval_metrics.py \
        --index data/index_val.jsonl \
        --pred_dir outputs/seine_val \
        --metrics psnr,ssim,lpips,clip,fvd \
        --output outputs/seine_val/metrics.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_rgb(path, size=None):
    """Load PNG as HWC uint8; optionally resize to (w, h)."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        if size is not None and img.size != size:
            img = img.resize(size, Image.BICUBIC)
        return np.array(img, dtype=np.uint8)


def collect_pair(row, pred_dir):
    """Return aligned (pred_frames, gt_frames) lists of HWC uint8 at GT resolution.

    Predictions are resized to the GT frame size so per-pixel metrics are
    computed on a common grid (GT defines the reference resolution)."""
    sample_id = row["sample_id"]
    gt_paths = row.get("future_frame_paths") or []
    if not gt_paths:
        return None, None, "no GT future frames in index (need val split)"

    case_dir = Path(pred_dir) / sample_id
    n = int(row["future_len"])
    pred_paths = [case_dir / f"{i}.png" for i in range(n)]
    missing = [p.name for p in pred_paths if not p.exists()]
    if missing:
        return None, None, f"missing {len(missing)} predicted frames (e.g. {missing[:3]})"

    t = min(len(gt_paths), len(pred_paths))
    if t < len(pred_paths) or t < len(gt_paths):
        # length mismatch is expected only if GT count != frame_length; align to min
        pass
    gt = [load_rgb(gt_paths[i]) for i in range(t)]
    gt_size = (gt[0].shape[1], gt[0].shape[0])  # (w, h)
    pred = [load_rgb(pred_paths[i], size=gt_size) for i in range(t)]
    return pred, gt, None


# --------------------------------------------------------------------------- #
# per-frame pixel/perceptual metrics
# --------------------------------------------------------------------------- #
def make_psnr_ssim():
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn

    def psnr(gt, pred):
        return float(psnr_fn(gt, pred, data_range=255))

    def ssim(gt, pred):
        return float(ssim_fn(gt, pred, channel_axis=2, data_range=255))

    return psnr, ssim


def make_lpips(net, device):
    import lpips as lpips_lib

    model = lpips_lib.LPIPS(net=net).to(device).eval()

    def to_tensor(arr):  # HWC uint8 -> 1,3,H,W in [-1,1]
        t = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        return t.to(device)

    @torch.no_grad()
    def lpips_dist(gt, pred):
        return float(model(to_tensor(gt), to_tensor(pred)).item())

    return lpips_dist


def make_clip(model_name, pretrained, device):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def clip_score(frames_uint8, text):
        imgs = torch.stack([preprocess(Image.fromarray(f)) for f in frames_uint8]).to(device)
        tok = tokenizer([text]).to(device)
        img_emb = model.encode_image(imgs)
        txt_emb = model.encode_text(tok)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        cos = (img_emb @ txt_emb.T).squeeze(1)  # (T,)
        return float((cos.clamp(min=0).mean() * 100).item())

    return clip_score


# --------------------------------------------------------------------------- #
# FVD (set-level)
# --------------------------------------------------------------------------- #
def sample_video(frames, seq_len, resolution):
    """frames: list HWC uint8 -> (seq_len, R, R, 3) uint8 via uniform temporal sampling."""
    n = len(frames)
    idx = np.linspace(0, n - 1, num=seq_len).round().astype(int)
    out = []
    for i in idx:
        img = Image.fromarray(frames[i]).resize((resolution, resolution), Image.BICUBIC)
        out.append(np.array(img, dtype=np.uint8))
    return np.stack(out, axis=0)


def compute_fvd(pred_videos, gt_videos, model_name, seq_len, resolution, device):
    from cdfvd import fvd

    pred_arr = np.stack([sample_video(v, seq_len, resolution) for v in pred_videos])  # B,T,R,R,3
    gt_arr = np.stack([sample_video(v, seq_len, resolution) for v in gt_videos])

    evaluator = fvd.cdfvd(model_name, n_real="full", n_fake="full", device=device)
    real_loader = evaluator.load_videos(gt_arr, data_type="video_numpy")
    evaluator.compute_real_stats(real_loader)
    fake_loader = evaluator.load_videos(pred_arr, data_type="video_numpy")
    evaluator.compute_fake_stats(fake_loader)
    return float(evaluator.compute_fvd_from_stats())


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Track 5 offline metrics (PSNR/SSIM/LPIPS/CLIP-S/FVD).")
    ap.add_argument("--index", required=True, help="val index with GT future_frame_paths")
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--metrics", default="psnr,ssim,lpips,clip,fvd")
    ap.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    ap.add_argument("--clip_model", default="ViT-B-32")
    ap.add_argument("--clip_pretrained", default="openai")
    ap.add_argument("--clip_text", default="captions", choices=["captions", "prompt"],
                    help="'captions' = raw pedestrian+vehicle text; 'prompt' = built prompt")
    ap.add_argument("--fvd_model", default="i3d", choices=["i3d", "videomae"])
    ap.add_argument("--fvd_seq_len", type=int, default=16)
    ap.add_argument("--fvd_resolution", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = {m.strip() for m in args.metrics.split(",") if m.strip()}
    rows = load_jsonl(args.index)
    if args.limit:
        rows = rows[: args.limit]

    psnr_fn = ssim_fn = lpips_fn = clip_fn = None
    if "psnr" in metrics or "ssim" in metrics:
        psnr_fn, ssim_fn = make_psnr_ssim()
    if "lpips" in metrics:
        lpips_fn = make_lpips(args.lpips_net, device)
    if "clip" in metrics:
        clip_fn = make_clip(args.clip_model, args.clip_pretrained, device)

    acc = {k: [] for k in ("psnr", "ssim", "lpips", "clip")}
    pred_videos, gt_videos = [], []
    n_eval, skipped = 0, []

    for i, row in enumerate(rows, 1):
        pred, gt, err = collect_pair(row, args.pred_dir)
        if err:
            skipped.append((row["sample_id"], err))
            continue
        n_eval += 1

        if psnr_fn and "psnr" in metrics:
            acc["psnr"].append(np.mean([psnr_fn(g, p) for g, p in zip(gt, pred)]))
        if ssim_fn and "ssim" in metrics:
            acc["ssim"].append(np.mean([ssim_fn(g, p) for g, p in zip(gt, pred)]))
        if lpips_fn:
            acc["lpips"].append(np.mean([lpips_fn(g, p) for g, p in zip(gt, pred)]))
        if clip_fn:
            text = (row["caption_pedestrian"] + " " + row["caption_vehicle"]).strip() \
                if args.clip_text == "captions" else row["prompt"]
            acc["clip"].append(clip_fn(pred, text))
        if "fvd" in metrics:
            pred_videos.append(pred)
            gt_videos.append(gt)
        print(f"[{i}/{len(rows)}] {row['sample_id']}: ok")

    results = {}
    for k in ("psnr", "ssim", "lpips", "clip"):
        if acc[k]:
            results[{"clip": "clip_s"}.get(k, k)] = round(float(np.mean(acc[k])), 5)

    if "fvd" in metrics and len(pred_videos) >= 2:
        print(f"computing FVD ({args.fvd_model}) over {len(pred_videos)} videos ...")
        try:
            results["fvd"] = round(compute_fvd(pred_videos, gt_videos, args.fvd_model,
                                               args.fvd_seq_len, args.fvd_resolution, device), 4)
        except Exception as exc:  # noqa: BLE001
            results["fvd"] = None
            print(f"FVD failed ({exc}); check cd-fvd version / load_videos signature.")

    summary = {
        "num_evaluated": n_eval,
        "num_skipped": len(skipped),
        "metrics": results,
    }
    print("\n==== Track 5 metrics ====")
    print(f"evaluated {n_eval} samples, skipped {len(skipped)}")
    for k, v in results.items():
        arrow = "(higher better)" if k in ("psnr", "ssim", "clip_s") else "(lower better)"
        print(f"  {k:7s}: {v}  {arrow}")
    if skipped:
        print(f"skipped examples: {skipped[:5]}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
