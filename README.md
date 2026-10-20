# group_24_code

Code for the CA6127 Transformer and Generative Adversarial Networks group project, Group 24:
**Quality, Speed and Diversity in Few-Step Text-to-Image Generation: An Empirical Comparison of
Distilled SDXL Models.**

We compare SDXL with DPM-Solver++ (the multi-step baseline), LCM-LoRA, SDXL-Turbo and
SDXL-Lightning on image quality, text alignment, seed diversity and end-to-end latency, and
test a two-stage hybrid (a 1-step SDXL-Turbo draft refined by SDXL image-to-image). All
models are pretrained; nothing is trained. Datasets, model weights and generated images are
not included: models are downloaded from the Hugging Face Hub at pinned revisions, and the
data scripts download MS-COCO and PartiPrompts.

## Experiments

| ID | Question | Split |
| --- | --- | --- |
| E1 | Main comparison of every method's frozen configurations: FID, CLIP Score, ImageReward, latency, memory | evaluation (2,000 COCO prompts) |
| E2 | Per-method sweep over steps and guidance; the configuration-selection rule picks the E1 points | tuning (500 COCO prompts) |
| E3 | LCM-LoRA adapter scale | tuning, plus two points on evaluation |
| E4 | Two-stage hybrid: SDXL-Turbo draft, bicubic upscaling, SDXL img2img refinement | tuning for the strength, evaluation for the comparison |
| E5 | Hard prompts from PartiPrompts (quantity, text, positions and properties), with blind human judging | 87 frozen prompts |
| Diversity | LPIPS across 8 seeds at each method's quality point | 100 prompts from evaluation |

A timestep-spacing check (SDXL's default "leading" spacing against the "trailing" spacing
used by the baseline) and paired bootstrap intervals complement E2.

## Layout

```
configs/   one YAML file per experiment and method; frozen selections (see configs/README.md)
src/       data preparation, generation, scoring, timing, selection, tables and figures
roles/c/   workflow for the SDXL-Turbo/SDXL-Lightning sweeps, freezing, E1 and the diversity runs
slurm/     SLURM job scripts and step lists (slurm/chains/) for the GPU cluster
tests/     CPU-only unit tests
docs/      E5 judging guide
data/      prompt splits and FID reference images (created by src.prepare_data, not included)
outputs/   images, metrics, tables and figures (created by the scripts, not included)
```

## Environment

```bash
conda env create -f environment.yml
conda activate ca6127
```

All package versions in `environment.yml` are pinned (Python 3.10). torch is the CUDA 12.8
build, and transformers must stay at 4.x because ImageReward is not compatible with 5.x.
`python -m src.check_env` checks the GPU, the imports and access to the Hugging Face Hub.

## Data

```bash
python -m src.prepare_data
```

This downloads the MS-COCO 2017 validation captions and images and writes, under `data/`,
the disjoint tuning (500) and evaluation (2,000) splits (the first caption of each image),
the 100-prompt diversity subset and centre-cropped 512x512 FID reference images for the
tuning and evaluation splits. The split seed is fixed, so the splits are
reproducible. The frozen E5 prompts are included in `configs/e5/parti_prompts.jsonl`,
together with the review sheet `configs/e5/parti_review.csv` that records every decision.
`python -m src.prepare_parti freeze` rebuilds the prompt list from the review sheet;
`review` writes a fresh, undecided sheet from the pinned PartiPrompts file and refuses to
overwrite an existing one.

## Running

Every run is described by a config file, and every generation, scoring and timing script
takes `--config`:

```bash
python -m src.generate --config configs/e1/sdxl_base.yaml          # images
python -m src.evaluate --config configs/e1/sdxl_base.yaml --metrics clip,imagereward,fid
python -m src.timing   --config configs/e1/sdxl_base.yaml          # latency and peak memory
```

Generation resumes where it stopped and skips existing images. Each output folder
`outputs/<experiment>/<split>/<method>/<name>/` keeps the config, library versions, GPU,
model revisions, scheduler settings and a per-image log of UNet calls and timesteps; a run
is refused if the folder was produced with different settings. Metrics are appended to
`outputs/metrics/metrics.csv`, and per-image scores are written to `per_image_scores.jsonl`
in each folder. Latency is always measured with `src.timing` (`src.hybrid --timing` for E4)
on one GPU type, an NVIDIA L40S in our runs: submit timing jobs with `--nodelist=<node>`
(see `slurm/job.sbatch`), because `src.make_tables` uses the newest timing row whatever its
`hardware` column says. Generation times in the logs include no warm-up and are not used.

| Experiment | Configs | How to run |
| --- | --- | --- |
| E2 | `configs/e2/sdxl_base.yaml`, `lcm_lora.yaml`, `sdxl_turbo.yaml`, `sdxl_lightning.yaml` (`sdxl_lightning_1step.yaml` is exploratory) | `src.generate`, then `src.evaluate --metrics clip,imagereward`, then `python -m src.select_configs --config <file>`; latency with `src.timing --config <file>`; SDXL-Turbo and SDXL-Lightning through `roles.c.workflow e2 --method sdxl_turbo` (or `sdxl_lightning`; `slurm/chains/c_tuning.txt`) |
| Selection | `configs/e1/selection_<method>.json` | records written by `src.select_configs` to `outputs/selection/`, with every candidate's scores, and copied to `configs/e1/` when the E1 configs are frozen; `roles.c.workflow freeze --selection <record>` writes the per-point configs in `configs/c_frozen/` (called by `e2` for SDXL-Turbo and SDXL-Lightning; steps 5–6 of `slurm/chains/c_final_all.txt` for the baseline and LCM-LoRA) |
| E1 | `configs/e1/<method>.yaml`; `sdxl_base_8step.yaml` (latency-matched reference) and `sdxl_turbo_1024.yaml` (exploratory) | `src.generate`, `src.evaluate --metrics clip,imagereward,fid`, `src.timing`; SDXL-Turbo and SDXL-Lightning through `roles.c.workflow e1 --method <method>` (`slurm/chains/c_final_all.txt`) |
| Spacing check | `configs/e2/sdxl_base_spacing.yaml` | `slurm/chains/e2_spacing.txt`, then `src.bootstrap_ci` against the matching E2 points |
| E3 | `configs/e3/lcm_lora.yaml` (tuning), `eval_lcm_lora.yaml` (evaluation) | unfused LCM-LoRA at 8 steps and guidance 2.0; `src.generate`, then `src.evaluate --metrics clip,imagereward` (tuning) or `clip,imagereward,fid` (evaluation). The two evaluation scales are the selection-rule point and the highest-CLIP point of the tuning sweep, recorded in `configs/e3/selection_lcm_lora.json` (this record is not written by `src.select_configs`) |
| E4 | `configs/e4/hybrid.yaml`, `draft_only.yaml` (tuning); `eval_hybrid.yaml`, `eval_draft_only.yaml` (evaluation) | `python -m src.hybrid --config <file>`, then `src.evaluate` (`fid` added on evaluation); the strength is chosen with `src.select_configs --config configs/e4/hybrid.yaml` (`configs/e4/selection_hybrid.json`); latency with `src.hybrid --timing` on `hybrid.yaml` and `draft_only.yaml`; `slurm/chains/e4_hybrid.txt`, `e4_eval.txt` |
| E5 | `configs/e5/<method>.yaml` | `slurm/chains/e5_parti.txt`; human judging with `src.e5_judging` ([docs/e5_judging.md](docs/e5_judging.md)); the judges' returned sheets are not included, so failure rates need a new judging round |
| Diversity | `configs/c_frozen/diversity/<method>.yaml` | `roles.c.workflow diversity --method <method>` (generates, then scores LPIPS, CLIP Score and ImageReward; `slurm/chains/c_final_all.txt`) |

Field-by-field documentation of the config files, the grid syntax and the automatic checks
is in [configs/README.md](configs/README.md).

### Selection rule

`src.select_configs` reads the newest complete scores of an E2 tuning grid and picks the
**quality point** (the highest ImageReward among the points whose CLIP Score is within 1.0
of the method's best) and the **lowest-step point** (the fewest steps with the quality
point's other settings). Files marked `exploratory: true` are never selected. The frozen E1
configs are written from these records before any evaluation-split image is generated.

### Tables, figures and intervals

```bash
python -m src.make_tables outputs/metrics/metrics.csv [more metrics.csv ...] --out outputs/tables
python -m src.plot_results --tables outputs/tables --out outputs/figures
python -m src.bootstrap_ci --ref base50=<folder>/per_image_scores.jsonl other=<folder>/per_image_scores.jsonl
```

`src.make_tables` merges the metrics files (keeping the newest row per config and metric)
and writes the E1, latency, E2, E3, E4, diversity and E5 tables; `src.plot_results` draws
the report figures from them; `src.bootstrap_ci` gives paired bootstrap 95% intervals for
CLIP Score and ImageReward. The intervals in the report compare, with paths relative to
`outputs/`: `e1/eval/sdxl_base/steps8_cfg7.5`, `steps4_cfg7.5` and
`e1/eval/sdxl_turbo/steps{4,1}_cfg0_res1024` against `e1/eval/sdxl_base/steps50_cfg7.5`;
`e2/tune/sdxl_base/steps25_cfg7.5` against `e2/tune/sdxl_base/steps50_cfg7.5`; and
`e2_spacing/tune/sdxl_base/steps{4,8,15}_cfg7.5_timestep_spacingleading` against
`e2/tune/sdxl_base/steps{4,8,15}_cfg7.5`, for example

```bash
python -m src.bootstrap_ci --ref trailing4=outputs/e2/tune/sdxl_base/steps4_cfg7.5/per_image_scores.jsonl \
    leading4=outputs/e2_spacing/tune/sdxl_base/steps4_cfg7.5_timestep_spacingleading/per_image_scores.jsonl
```

`roles.c.summarize` writes completion checks, sweep plots and diversity grids to
`outputs/c_handoff/`.

### Tests

```bash
python -m unittest tests.test_bootstrap_ci tests.test_e5_judging roles.c.test_workflow
```

The tests need no GPU and download no models; they work in temporary folders under
`outputs/test_tmp/`.

## Running on a SLURM cluster

The scripts in `slurm/` were written for the course GPU cluster (partition `MGPU-TC2`;
change the `#SBATCH` lines for another cluster). There, a job may run for at most 6 hours,
a user may run 2 jobs at a time, and compute nodes cannot submit jobs. All commands are
submitted from the repository root on the login node:

```bash
bash slurm/setup_env.sh                                       # once: conda environment and logs/
sbatch --job-name=data slurm/job.sbatch src.prepare_data       # once: splits and FID references
sbatch slurm/smoke.sbatch                                      # 10 images per method, then scoring

# One module per job
sbatch --job-name=eval slurm/job.sbatch src.evaluate --config configs/e2/sdxl_base.yaml --metrics clip,imagereward

# Longer pipelines: a step list run inside one job, split into jobs under 6 hours
J1=$(sbatch --parsable --job-name=e2-base slurm/chain.sbatch slurm/chains/e2_sdxl_base.txt 1 2)
sbatch --dependency=afterok:$J1 --kill-on-invalid-dep=yes --job-name=e2-base \
    slurm/chain.sbatch slurm/chains/e2_sdxl_base.txt 3 4
```

Each line of a step list is `<module> <arguments>`; `chain.sbatch <list> <from> <to>` runs
the steps from `<from>` to `<to>` in order and stops at the first failure, which also
cancels a dependent job. Resubmit from the failed step; generation continues where it
stopped.
