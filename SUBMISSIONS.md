# Track 5 — Submission log

Naming on the site (max 15 chars): `v{N}-{sampler}{steps}-cfg{X}` (add `-k{K}` or
`-r{H}` when those change). Full config + git commit + score recorded here.

| Name | Date | Commit | Sampler | cfg | cond_frames | image_size | Notes | Public score |
|------|------|--------|---------|-----|-------------|------------|-------|--------------|
| `v1-d30-cfg5` | 2026-07-02 | 4134684 | ddim 30 | 5.0 | 2 | [288,512] | first valid submission (draft quality) | TBD |

## Config → knob legend
- `d30` = `sample_method: ddim`, `num_sampling_steps: 30`
- `cfg5` = `cfg_scale: 5.0`
- `k2` = `cond_frames: 2`
- `r{H}` = generation height in `image_size: [H, W]` (W follows 16:9)

## How to reproduce a row
`git checkout <commit>` → the exact `configs/seine_track5.yaml` for that submission
is what produced it. Record the Public score once the site reports it.
