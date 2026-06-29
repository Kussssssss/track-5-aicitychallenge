# SEINE inference for AI City 2026 Track 5 (Colab A100)

Inference-only pipeline: pretrained SEINE (`seine.pt`) conditioned on the last
history frame(s) via `mask_type=first{K}`, autoregressively rolled out to the
required `frame length` (N), then resized to each sample's native resolution and
written as `0.png .. (N-1).png`.

> **One-click path:** open `notebooks/track5_seine_colab.ipynb` in Colab (GPU
> runtime), edit the CONFIG cell, and run top to bottom. The steps below are the
> same commands explained in detail.

## Pipeline overview

```
WTS test data ──► data/build_index.py ──► index_test.jsonl
                                                │
                                                ▼
                                    baselines/seine_infer.py  (SEINE)
                                                │
                                                ▼
                                    outputs/seine/<sample>/0.png..
                                                │
              submission/validate_submission.py ─┴─► submission/make_zip.py ─► .zip
```

The only new model component is `baselines/seine_infer.py` + `configs/seine_track5.yaml`;
indexing / validation / zipping are reused from the existing repo.

## Step 0 — Colab setup (A100)

```bash
from google.colab import drive
drive.mount('/content/drive')
```

```bash
# repos
%cd /content
!git clone <your_aicity_track5_repo> aicity_track5      # or copy from Drive
!git clone https://github.com/Vchitect/SEINE third_party/SEINE

%cd /content/aicity_track5
!pip install -q omegaconf einops natsort decord imageio timm rotary-embedding-torch Pillow
# Keep diffusers pinned (SEINE uses its ~0.15 APIs); use a recent transformers so
# tokenizers installs from a prebuilt wheel. Do NOT pin transformers==4.29.2 -- it
# drags in an old tokenizers that must build from Rust and fails on Colab py3.12.
!pip install -q "huggingface_hub==0.25.2" "diffusers==0.15.0" "transformers==4.41.2"
# diffusers 0.15 imports flax modules (using the removed jax.random.KeyArray) when
# jax+flax are present; we don't use flax, so remove it to skip that import path.
!pip uninstall -y flax
# Colab already ships a recent torch/torchvision; only install xformers if the
# wheel matches the torch build, otherwise set enable_xformers... : False in the config.
```

> If you already triggered the `jax.random.KeyArray` error, **Runtime > Restart
> session** after uninstalling flax (the failed import leaves diffusers half-loaded
> in the kernel), then re-run from the import check.

Quick import sanity check before the long run:

```python
import diffusers, transformers
from diffusers.models.attention import FeedForward, AdaLayerNorm
from diffusers.utils import WEIGHTS_NAME
from transformers import CLIPTokenizer, CLIPTextModel
print(diffusers.__version__, transformers.__version__, "OK")
```

### Download weights into pretrained/

```bash
%cd /content/aicity_track5
!mkdir -p pretrained
# SEINE checkpoint (google drive folder 1cWfeDzKJhpb0m6HA5DoMOH0_ItuUY95b)
!pip install -q gdown
!gdown --fuzzy "https://drive.google.com/drive/folders/1cWfeDzKJhpb0m6HA5DoMOH0_ItuUY95b" -O pretrained --folder
# Stable Diffusion v1-4 base (VAE + UNet config + tokenizer/text encoder)
!pip install -q "huggingface_hub"
!huggingface-cli download CompVis/stable-diffusion-v1-4 --local-dir pretrained/stable-diffusion-v1-4
```

Expected:
```
pretrained/
├── seine.pt
└── stable-diffusion-v1-4/{vae, unet, text_encoder, tokenizer, ...}
```

## Step 0b — Download the dataset

The **test** set is a single pre-staged `.zip` (`<sample>/input/%d.png + caption.json`):

```bash
!pip install -q gdown
!gdown 1TJcgXk7RRkHB7JjN7uWVHuTKLgzmtfhq -O /content/data/wts_test.zip
!unzip -q -o /content/data/wts_test.zip -d /content/data/test
```

Use `/content/data/test` as `data_root` — `build_index.py`/`inspect_dataset.py` scan
for `input/` recursively, so a wrapper folder inside the zip is fine.

The **train/val** set is *raw video* (`videos/`, `annotations/`, `external/BDD_PC_5K/`)
and must be staged into `<sample>/input/*.png` + GT with
`wts-dataset-tv2v/script/data_sample.py` before `eval/eval_metrics.py` can score it —
not needed for the test-submission path.

```bash
# optional, large:
# !gdown --folder 1d7PHfIOcE9UClirWKbsoZKWK8ceDpjKU -O /content/data/wts_trainval
```

## Step 1 — Inspect the real test layout (IMPORTANT)

The index builder assumes `caption.json` sits **next to** an `input/` folder.
Confirm this against the actual download before trusting the index:

```bash
!python data/inspect_dataset.py --data_root "/content/drive/MyDrive/<wts_test>"
```

Check: where `caption.json` lives, how many `input/*.png` history frames each
sample has (sets a sensible `cond_frames`), the frame resolution, and the
distribution of `frame length` (N). If the layout differs, adjust
`data/build_index.py` (`discover_sample_dirs`) accordingly.

## Step 2 — Build the index

```bash
# test (no GT) -- for the actual submission
!python data/build_index.py \
    --data_root "/content/drive/MyDrive/<wts_test>" \
    --split test --output data/index_test.jsonl

# val (has GT) -- for offline metric tuning
!python data/build_index.py \
    --data_root "/content/drive/MyDrive/<wts_val>" \
    --split val --output data/index_val.jsonl
```

## Step 3 — Smoke test SEINE on a few samples

```bash
!python baselines/seine_infer.py \
    --config configs/seine_track5.yaml \
    --index data/index_test.jsonl \
    --output outputs/seine \
    --seine_root /content/third_party/SEINE \
    --limit 2
```

Then sanity-check the format:

```bash
!python submission/validate_submission.py \
    --index data/index_test.jsonl --pred_dir outputs/seine
```

(Validation will complain about the not-yet-generated samples; with `--limit`
that is expected — only check the two generated folders look right.)

## Step 4 — Evaluate + tune on val (offline metrics)

The competition scores PSNR, SSIM, LPIPS, CLIP-S, FVD. The val split has GT, so
generate predictions on it and score them with `eval/eval_metrics.py`.

```bash
!pip install -q -r requirements-eval.txt   # scikit-image, lpips, open_clip_torch, cd-fvd

# 1) generate predictions on val
!python baselines/seine_infer.py \
    --config configs/seine_track5.yaml \
    --index data/index_val.jsonl \
    --output outputs/seine_val \
    --seine_root /content/third_party/SEINE

# 2) score against GT (future_frame_paths recorded by build_index)
!python eval/eval_metrics.py \
    --index data/index_val.jsonl \
    --pred_dir outputs/seine_val \
    --metrics psnr,ssim,lpips,clip,fvd \
    --output outputs/seine_val/metrics.json
```

Then sweep the knobs in `configs/seine_track5.yaml` and re-score:

- `image_size`: `[320, 512]` vs `[288, 512]` (true 16:9) vs `[240, 560]`.
- `cond_frames` (K): 1 (I2V, least drift) vs 2–4 (more motion context).
- `cfg_scale`: 6–9. `num_sampling_steps` / `sample_method` (`ddim` 50–100 is much faster).
- `additional_prompt` / `negative_prompt` wording.

Metric notes (implementations are standard; the official script is not public):
- **CLIP-S** = CLIP image-text cosine ×100, generated frame vs the raw
  pedestrian+vehicle captions (`--clip_text captions`, default). Switch to the
  built prompt with `--clip_text prompt`.
- **FVD** uses `cd-fvd` with the I3D backbone (`--fvd_model i3d`), computed over
  the whole val set (needs ≥2 samples). Videos are uniformly sampled to
  `--fvd_seq_len` frames at `--fvd_resolution`.
- Run a quick subset first with `--limit 10` to validate wiring before the full pass.

## Step 5 — Full test run + submission

```bash
!python baselines/seine_infer.py \
    --config configs/seine_track5.yaml \
    --index data/index_test.jsonl \
    --output outputs/seine \
    --seine_root /content/third_party/SEINE
# resumable: re-running skips samples that already have N frames (omit --overwrite)

!python submission/validate_submission.py \
    --index data/index_test.jsonl --pred_dir outputs/seine

!python submission/make_zip.py \
    --pred_dir outputs/seine --zip_path outputs/submission_seine.zip
```

## Known risk areas (where quality is won or lost)

1. **Autoregressive drift** — when N >> (num_frames − K), errors compound across
   windows. Mitigate with small K, more sampling steps, or a more constrained prompt.
2. **Resolution** — SEINE generates on a small canvas then upscales to 1280×720;
   pick `image_size` close to 16:9 to minimise distortion.
3. **Motion speed / fps** — the temporal stride of the GT N frames is unknown;
   SEINE's intrinsic motion rate may not match. Tune on val.
4. **Data layout** — verify `caption.json` placement (Step 1) before trusting the index.
