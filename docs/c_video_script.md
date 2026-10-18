# Role C recording script (about 2–3 minutes)

My part evaluates SDXL-Turbo, SDXL-Lightning, and diversity across random seeds. We use released pretrained models and the shared generation and evaluation scripts.

Turbo is evaluated at one to four steps and 512-pixel generation. Lightning uses separate two-, four-, and eight-step checkpoints at 1024 pixels. Its experimental one-step checkpoint is reported separately.

We choose configurations on 500 tuning prompts. ImageReward determines the quality point, provided CLIP is within one point of the method's best score. This selects four steps for both Turbo and Lightning. The minimum-step points are one step for Turbo and two for Lightning. Lightning's eight-step checkpoint does not outperform four steps on our tuning metrics, so additional checkpoint steps are not consistently better in this experiment.

We then freeze those configurations and evaluate them on 2,000 separate prompts. The table shows the actual quality and latency tradeoffs. These evaluation results are not used to retune the models.

For diversity, every method generates eight seeds for the same 100 prompts. We average all 28 seed pairs within each prompt before averaging across prompts. The baseline uses 50 steps and guidance 7.5, while LCM uses B's frozen eight-step configuration.

The largest observed mean LPIPS is 0.6747 for LCM-LoRA. This measures dissimilarity, which can also come from errors. We therefore inspect CLIP, ImageReward and the fixed-random image grid together. Native generation resolutions differ, and our inference comparison cannot identify the causal effect of individual training losses.

Recording cue: show the measured E1 and diversity tables and the fixed-random example. Adjust pacing to D's allotted duration.
