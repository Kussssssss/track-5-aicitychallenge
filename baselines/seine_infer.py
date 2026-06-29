"""SEINE inference adapter for AI City 2026 Track 5 (TV2V future prediction).

Reads the JSONL index produced by data/build_index.py and, for each sample,
conditions SEINE on the last `cond_frames` history frames (mask_type=first{K})
and autoregressively rolls out until `future_len` (N) frames are generated, then
resizes each frame back to the sample's native input resolution and writes them
as 0.png .. (N-1).png -- the official Track 5 submission layout.

Example (Colab, repo root = aicity_track5/):
    python baselines/seine_infer.py \
        --config configs/seine_track5.yaml \
        --index data/index_test.jsonl \
        --output outputs/seine \
        --seine_root ../third_party/SEINE
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


# --------------------------------------------------------------------------- #
# index / io helpers
# --------------------------------------------------------------------------- #
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_frame_uint8(path):
    """Load a PNG as an HWC uint8 RGB numpy array."""
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# SEINE setup
# --------------------------------------------------------------------------- #
def import_seine(seine_root):
    """Make the SEINE package importable and return the symbols we need."""
    seine_root = str(Path(seine_root).resolve())
    if seine_root not in sys.path:
        sys.path.insert(0, seine_root)
    from diffusers.models import AutoencoderKL
    from einops import rearrange
    from torchvision import transforms

    from datasets import video_transforms
    from diffusion import create_diffusion
    from models import get_models
    from models.clip import TextEmbedder
    from utils import mask_generation_before

    return dict(
        AutoencoderKL=AutoencoderKL,
        rearrange=rearrange,
        transforms=transforms,
        video_transforms=video_transforms,
        create_diffusion=create_diffusion,
        get_models=get_models,
        TextEmbedder=TextEmbedder,
        mask_generation_before=mask_generation_before,
    )


def build_args(cfg):
    """Flatten the OmegaConf into the attribute namespace SEINE expects."""
    args = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    h, w = int(args.image_size[0]), int(args.image_size[1])
    args.image_h, args.image_w = h, w
    args.latent_h, args.latent_w = h // 8, w // 8
    return args


def load_models(args, S, device):
    print("loading SEINE UNet ...")
    model = S["get_models"](args).to(device)
    if args.enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available

        if is_xformers_available():
            model.enable_xformers_memory_efficient_attention()
        else:
            print("warning: xformers unavailable, continuing without it")
    state = torch.load(args.ckpt, map_location=lambda s, _l: s)["ema"]
    model.load_state_dict(state)
    model.eval()

    diffusion = S["create_diffusion"](str(args.num_sampling_steps))
    vae = S["AutoencoderKL"].from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)
    text_encoder = S["TextEmbedder"](args.pretrained_model_path).to(device)
    if args.use_fp16:
        vae.to(dtype=torch.float16)
        model.to(dtype=torch.float16)
        text_encoder.to(dtype=torch.float16)
    print("SEINE ready")
    return model, vae, text_encoder, diffusion


# --------------------------------------------------------------------------- #
# core generation
# --------------------------------------------------------------------------- #
def make_transform(S, args):
    return S["transforms"].Compose([
        S["video_transforms"].ToTensorVideo(),
        S["video_transforms"].ResizeVideo((args.image_h, args.image_w)),
        S["transforms"].Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])


def frames_to_clip(cond_frames_uint8, num_frames, transform, device):
    """cond_frames_uint8: list of K HWC uint8 arrays -> (1, F, C, H, W) float clip."""
    tensors = [torch.as_tensor(f).unsqueeze(0) for f in cond_frames_uint8]
    zeros = torch.zeros_like(tensors[0])
    tensors += [zeros] * (num_frames - len(tensors))
    clip = torch.cat(tensors, dim=0).permute(0, 3, 1, 2)  # f, c, h, w (uint8)
    clip = transform(clip)  # f, c, h, w (float, normalized)
    return clip.unsqueeze(0).to(device)  # b=1, f, c, h, w


def auto_inpainting(args, masked_video, mask, prompt, vae, text_encoder, diffusion, model, device, S):
    """Adapted from SEINE/sample_scripts/with_mask_sample.py: returns (F,3,H,W) in [-1,1]."""
    rearrange = S["rearrange"]
    b, f, c, h, w = masked_video.shape
    dtype = torch.float16 if args.use_fp16 else torch.float32

    z = torch.randn(1, 4, args.num_frames, args.latent_h, args.latent_w, dtype=dtype, device=device)
    masked_video = masked_video.to(dtype=dtype)
    mask = mask.to(dtype=dtype)

    masked_video = rearrange(masked_video, "b f c h w -> (b f) c h w").contiguous()
    masked_video = vae.encode(masked_video).latent_dist.sample().mul_(0.18215)
    masked_video = rearrange(masked_video, "(b f) c h w -> b c f h w", b=b).contiguous()
    mask = torch.nn.functional.interpolate(mask[:, :, 0, :], size=(args.latent_h, args.latent_w)).unsqueeze(1)

    if args.do_classifier_free_guidance:
        masked_video = torch.cat([masked_video] * 2)
        mask = torch.cat([mask] * 2)
        z = torch.cat([z] * 2)
        prompt_all = [prompt, args.negative_prompt]
    else:
        prompt_all = [prompt]

    text_prompt = text_encoder(text_prompts=prompt_all, train=False)
    model_kwargs = dict(
        encoder_hidden_states=text_prompt,
        class_labels=None,
        cfg_scale=args.cfg_scale,
        use_fp16=args.use_fp16,
    )

    if args.sample_method == "ddim":
        sample_fn = diffusion.ddim_sample_loop
    else:
        sample_fn = diffusion.p_sample_loop
    samples = sample_fn(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs,
        progress=False, device=device, mask=mask, x_start=masked_video, use_concat=args.use_mask,
    )
    if args.do_classifier_free_guidance:
        samples, _ = samples.chunk(2, dim=0)

    video_clip = samples[0].permute(1, 0, 2, 3).contiguous()  # f, 4, lh, lw
    video_clip = vae.decode(video_clip / 0.18215).sample  # f, 3, H, W in [-1,1]
    return video_clip


def clip_to_uint8(video_clip):
    """(F,3,H,W) in [-1,1] -> list of HWC uint8 arrays."""
    arr = ((video_clip * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(torch.uint8)
    arr = arr.cpu().permute(0, 2, 3, 1).numpy()  # f, H, W, 3
    return [arr[i] for i in range(arr.shape[0])]


def generate_future(args, sample, models, transform, device, S):
    """Autoregressively roll out future_len frames at model resolution (uint8 HWC)."""
    model, vae, text_encoder, diffusion = models
    K = int(args.cond_frames)
    per_clip = args.num_frames - K  # future frames produced per clip
    if per_clip <= 0:
        raise ValueError(f"cond_frames ({K}) must be < num_frames ({args.num_frames})")

    prompt = sample["prompt"] + args.additional_prompt
    history = sample["history_frame_paths"]
    if not history:
        raise ValueError(f"{sample['sample_id']}: no history frames")

    # initial condition = last K history frames (resized to model canvas at load via transform)
    cond = [load_frame_uint8(p) for p in history[-K:]]
    # resize condition frames to model canvas so zeros-padding shapes match
    cond = [np.array(Image.fromarray(c).resize((args.image_w, args.image_h), Image.BICUBIC)) for c in cond]

    future = []
    n_target = int(sample["future_len"])
    while len(future) < n_target:
        clip = frames_to_clip(cond, args.num_frames, transform, device)
        mask = S["mask_generation_before"](f"first{K}", clip.shape, clip.dtype, device)
        masked_video = clip * (mask == 0)
        video_clip = auto_inpainting(args, masked_video, mask, prompt, vae, text_encoder,
                                     diffusion, model, device, S)
        gen = clip_to_uint8(video_clip)[K:]  # drop the K reconstructed condition frames
        future.extend(gen)
        cond = future[-K:]  # re-condition on the last K generated frames
    return future[:n_target]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="SEINE inference for AI City Track 5.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seine_root", default="../third_party/SEINE")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N samples (smoke test)")
    parser.add_argument("--overwrite", action="store_true")
    args_cli = parser.parse_args()

    cfg = OmegaConf.load(args_cli.config)
    args = build_args(cfg)
    if args.seed:
        torch.manual_seed(int(args.seed))
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device found; SEINE will be extremely slow on CPU.")

    S = import_seine(args_cli.seine_root)
    models = load_models(args, S, device)
    transform = make_transform(S, args)

    output_root = Path(args_cli.output)
    output_root.mkdir(parents=True, exist_ok=True)

    samples = list(load_jsonl(args_cli.index))
    if args_cli.limit:
        samples = samples[: args_cli.limit]

    for idx, sample in enumerate(samples, 1):
        sample_id = sample["sample_id"]
        case_dir = output_root / sample_id
        n_target = int(sample["future_len"])
        existing = sorted(case_dir.glob("*.png")) if case_dir.is_dir() else []
        if len(existing) >= n_target and not args_cli.overwrite:
            print(f"[{idx}/{len(samples)}] {sample_id}: already done, skip")
            continue
        case_dir.mkdir(parents=True, exist_ok=True)

        w, h = int(sample["width"]), int(sample["height"])
        print(f"[{idx}/{len(samples)}] {sample_id}: generating {n_target} frames -> {w}x{h}")
        future = generate_future(args, sample, models, transform, device, S)

        for i, frame in enumerate(future):
            img = Image.fromarray(frame)
            if img.size != (w, h):
                img = img.resize((w, h), Image.BICUBIC)
            img.save(case_dir / f"{i}.png")

    print(f"done. predictions at {output_root}")


if __name__ == "__main__":
    main()
