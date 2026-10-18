# Role C: adversarial distillation and diversity results

All numbers below are our measured results. Literature evidence is kept separately in [Methods and original-paper evidence](c_methods_and_evidence.md).

## Protocol

We used the shared COCO split (500 tuning prompts, 2,000 disjoint evaluation prompts, and 100 diversity prompts drawn from evaluation). All generation used fp16; metric images were 512 × 512. Turbo generated at 512 pixels, whereas Lightning, LCM-LoRA and the baseline generated at 1024 pixels. Downsampling can hide fine detail. These are comparisons of recommended deployment configurations. Training procedures, guidance and native resolutions differ, so inference comparisons cannot isolate the effect of a training loss. No model training was performed.

All C jobs ran on TC2N07 with NVIDIA L40S. Timings use A's script: batch size one, warm-up, synchronized text-encoding-to-VAE-decoding calls, and 20 measured runs. Latency is median seconds; the CSV also contains its IQR. A verifies cross-method consistency.

## E2: official tuning configurations

Each point uses 500 prompts and seed 0. Lightning changes checkpoint with step count.

| Method / configuration | Native px | Steps | CFG | clip_score | image_reward | latency_median_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SDXL-Turbo / steps1_cfg0 | 512 | 1 | 0 | 26.3775 | 0.6864 | 0.1121 |
| SDXL-Turbo / steps2_cfg0 | 512 | 2 | 0 | 26.1386 | 0.7174 | 0.1488 |
| SDXL-Turbo / steps3_cfg0 | 512 | 3 | 0 | 26.0867 | 0.7317 | 0.1855 |
| SDXL-Turbo / steps4_cfg0 | 512 | 4 | 0 | 26.0613 | 0.7397 | 0.2208 |
| SDXL-Lightning / ckpt2_cfg0 | 1024 | 2 | 0 | 25.903 | 0.5896 | 0.4067 |
| SDXL-Lightning / ckpt4_cfg0 | 1024 | 4 | 0 | 25.8769 | 0.6741 | 0.5309 |
| SDXL-Lightning / ckpt8_cfg0 | 1024 | 8 | 0 | 25.6112 | 0.6565 | 0.7798 |

### Frozen points

- SDXL-Turbo: quality point 4 steps; minimum-step point 1 step; guidance 0. The selected point is 0.3162 CLIP points below the method's best, within the preset 1.0-point tolerance. Both points were frozen before final evaluation.
- SDXL-Lightning: quality point 4 steps; minimum-step point 2 steps; guidance 0. The selected point is 0.0261 CLIP points below the method's best, within the preset 1.0-point tolerance. Both points were frozen before final evaluation.

Turbo's tuning ImageReward increases from 0.6864 at one step to 0.7397 at four steps, while CLIP decreases from 26.3775 to 26.0613. Lightning's four-step checkpoint scores higher than its eight-step checkpoint in both tuning ImageReward (0.6741 versus 0.6565) and CLIP (25.8769 versus 25.6112). Thus this checkpoint comparison does not show monotonic improvement with step count. These are observations on the fixed tuning subset.

## One-step Lightning: exploratory only

The x0 checkpoint uses sample prediction with a trailing Euler scheduler. It is excluded from quality-point selection and the main comparison.

| Method / configuration | Native px | Steps | CFG | clip_score | image_reward | latency_median_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SDXL-Lightning / ckpt1_exploratory | 1024 | 1 | 0 | 25.9447 | 0.479 | 0.345 |

## E1: frozen evaluation

Each configuration uses all 2,000 evaluation prompts and seed 0. The FID reference is the corresponding 2,000 real images, center-cropped and resized by the shared preparation script. Limited-sample FID is for within-project comparisons, not direct comparisons with paper-reported FID. Configurations were not retuned using this table.

| Method / configuration | Native px | Steps | CFG | clip_score | image_reward | fid | latency_median_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SDXL-Lightning / ckpt2_cfg0 | 1024 | 2 | 0 | 26.0576 | 0.6138 | 54.2174 | 0.4051 |
| SDXL-Lightning / ckpt4_cfg0 | 1024 | 4 | 0 | 26.035 | 0.6687 | 52.9391 | 0.5281 |
| SDXL-Turbo / steps1_cfg0 | 512 | 1 | 0 | 26.3743 | 0.672 | 49.1713 | 0.1119 |
| SDXL-Turbo / steps4_cfg0 | 512 | 4 | 0 | 26.0988 | 0.7546 | 48.7951 | 0.2161 |

For SDXL-Turbo, the tuning-selected quality point differs from the minimum-step point by +0.0826 ImageReward, -0.2755 CLIP points and -0.3762 FID on evaluation. Median latency is 0.2161 versus 0.1119 seconds.

For SDXL-Lightning, the tuning-selected quality point differs from the minimum-step point by +0.0549 ImageReward, -0.0226 CLIP points and -1.2783 FID on evaluation. Median latency is 0.5281 versus 0.4051 seconds.

## Seed diversity

Each row contains 100 identical prompts × seeds 0–7 (800 images). We compute all 28 seed pairs within a prompt, then average the 100 prompt means. LPIPS n=100 refers to prompts; CLIP and ImageReward n=800 refer to images. All methods use their frozen quality-priority point. The baseline is fixed at 50 steps and CFG 7.5 as specified by the group leader. LCM uses B's E2 selection: eight steps, guidance 2.0, fused LoRA at scale 1.0.

| Method / configuration | Native px | Steps | CFG | lpips_diversity | clip_score | image_reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LCM-LoRA / steps8_cfg2_scale1 | 1024 | 8 | 2 | 0.6747 | 26.5865 | 0.5941 |
| SDXL baseline / steps50_cfg7.5 | 1024 | 50 | 7.5 | 0.6693 | 26.8785 | 0.7645 |
| SDXL-Lightning / ckpt4_cfg0 | 1024 | 4 | 0 | 0.6453 | 26.328 | 0.6908 |
| SDXL-Turbo / steps4_cfg0 | 512 | 4 | 0 | 0.634 | 26.4351 | 0.8388 |

Mean LPIPS ranges from 0.6340 (SDXL-Turbo) to 0.6747 (LCM-LoRA). This describes seed dissimilarity on this subset. Greater dissimilarity can also include semantic errors or artifacts; interpret it with the alignment scores and fixed-random grids. These averages do not establish mode collapse or a causal effect of adversarial training. No statistical-significance claim is made.

## Reproduction and handoff

The measured CSV is included in the separate C handoff package for D to integrate.

C's official selection records and E1 grids are in `configs/e1/`; execution manifests, per-point YAMLs and split hashes are in `configs/c_frozen/`. Shared source modules were not changed. The complete measured CSV, status audit, score plots, diversity grids and a fixed-random prompt with 32 source images are in the separate `outputs/c_handoff/` handoff package, which is Git-ignored. Runtime environment and SLURM logs are retained in the cluster account. Data, model weights, images and secrets are not committed.

Core jobs: data 43639; four-method smoke 43640; C tuning 43641; final evaluation/diversity 43644.

