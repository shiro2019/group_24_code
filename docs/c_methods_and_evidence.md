# Role C: report methods and literature evidence

## Methods text (English; ready to integrate)

SDXL-Turbo uses Adversarial Diffusion Distillation (ADD), combining a pretrained diffusion teacher with an adversarial objective. We evaluate its released checkpoint at 1, 2, 3 and 4 steps, with classifier-free guidance disabled and a generation resolution of 512 pixels. These are inference experiments; we do not retrain the student or discriminator. [ADD paper](https://arxiv.org/html/2311.17042v1); [official Turbo model card](https://huggingface.co/stabilityai/sdxl-turbo).

SDXL-Lightning combines progressive distillation and adversarial learning. We load a separate full-UNet checkpoint for each of 2, 4 and 8 steps, with an Euler scheduler using trailing timesteps, guidance zero, and 1024-pixel generation. The experimental one-step x0 checkpoint uses sample prediction and is reported separately. This comparison changes both checkpoint and step count. [Lightning paper](https://arxiv.org/html/2402.13929v2); [official Lightning model card](https://huggingface.co/ByteDance/SDXL-Lightning).

## Literature evidence: NOT our measured results

| Original evidence | Observation reported by authors | Boundary for our report |
| --- | --- | --- |
| ADD, Table 1(c), student initialization | Random/pretrained initialization: FID 293.6/20.6 | Small-backbone ablation; does not measure our released Turbo checkpoint |
| ADD, Table 1(d), loss terms | Adversarial-only/distillation-only/combined exponential: FID 20.8/315.6/20.6 | Supports combining objectives in that setup; not proof that our images improve for the same reason |
| Lightning, Section 3.4 and Figure 2 | Relaxing flow conditioning addresses Janus artifacts | Author mechanism evidence; no controlled retraining by our group |
| Lightning, Section 5.2, Figure 6 | Non-square inference has more failures at very low steps | Qualitative generalization evidence; our square-image protocol does not test aspect ratios |

ADD Table 1 uses COCO FID5k, 512px and short ablation training. Its default student is a smaller SD2.1-based model. Its CLIP implementation differs from ours. Do not compare these FIDs to our 2,000-sample FID, and do not present the table as SDXL-Turbo component ablations reproduced by us. [ADD Section 4.1](https://arxiv.org/html/2311.17042v1#S4.SS1).

Lightning's conditional/unconditional training discussion is not a numerical controlled ablation table. Label it as mechanism discussion and qualitative evidence. [Lightning Section 3.4](https://arxiv.org/html/2402.13929v2#S3.SS4).

## Our experiment protocol (English)

We use the same 500-prompt tuning set and seed 0 for all E2 candidates. The quality-priority configuration maximizes tuning ImageReward subject to a CLIP Score no more than 1.0 point below the method's best CLIP Score. We freeze that point and the official minimum-step point before using the 2,000-prompt evaluation set. Lightning's one-step checkpoint is excluded from selection.

For diversity, we use the quality-priority point of each method, the same 100 prompts drawn from the evaluation split, and eight seeds per prompt. LPIPS is computed between all 28 seed pairs for each prompt and then averaged over prompts. All metric images are 512 by 512 pixels. CLIP Score, ImageReward and fixed-random-sample grids accompany LPIPS, because image dissimilarity alone does not establish useful diversity. We do not infer mode collapse from a single metric or attribute cross-model differences to individual training losses.

Latency comparisons use A's common-GPU measurements with warm-up, batch size one, fp16 and synchronized end-to-end timing. Native generation resolutions differ and must appear beside each method in the result table.

## Results status and writing rules

No group experiment numbers are available in this document yet. Populate the final analysis from full metric CSV rows and frozen configurations after GPU jobs succeed. Report unsuccessful or incomplete runs explicitly.

Required analysis: identify the tuning-selected points, quantify quality changes across the Turbo steps and Lightning checkpoints, compare per-prompt seed diversity together with text alignment, and state the native-resolution and sample-size limitations. Describe the sign and size of observed differences; do not write a winner or failure explanation in advance.
