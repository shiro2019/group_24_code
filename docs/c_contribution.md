# Role C contribution

Contributor: Role C (adian-mars)

- Implemented Turbo 1/2/3/4-step and Lightning 2/4/8-checkpoint tuning configurations, plus a separate experimental one-step x0 configuration.
- Ran the shared four-method smoke test (10 images per method with CLIP and ImageReward), then an additional two-image x0 check. Six CPU-only freeze/split safety tests passed locally and on the compute node.
- Generated and evaluated 15,200 saved experiment images across official/exploratory tuning, frozen C-method E1, and four-method diversity. The 42 smoke images and unsaved timing calls are excluded from that count.
- Selected and froze Turbo 4/1-step and Lightning 4/2-step quality/minimum points using tuning metrics only; submitted selection records and E1 YAMLs in the group's required format.
- Ran 100 prompts × 8 seeds for SDXL baseline, LCM-LoRA, Turbo and Lightning, using baseline 50 steps/CFG 7.5 and B's official fused LCM selection.
- Produced measured metric tables, fixed-random comparison grids, report methods/results, original-paper evidence notes, contribution text and presentation material.
- Used A's generation, scoring and timing modules without modifying shared `src/`. B's E2 record and attached E3 package were reviewed as dependencies, not claimed as C's work. D owns final report/video integration and E5 annotation organization.

No model training or training-loss ablation was performed. Training-mechanism statements are attributed to original papers.
