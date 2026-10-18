# C 对组内依赖的核对

- 组长指定：多样性基线为 SDXL 50 步、CFG 7.5。来源为 main 的 `configs/e1/selection_sdxl_base.json` 的 quality 点。
- B 已发布 `b/e2-lcm-freeze` 分支，尚未出现在此次同步的 main；C 读取其选择记录作为输入，不合并或改动 B 的实现。来源 commit 记录在 `configs/c_inputs/provenance.json`。
- B 的 E2 质量点为 LCM 8 步、guidance 2.0、`lora_fuse: true`、`lora_scale: 1.0`。正式多样性采用它。
- 用户另提供 `b_lcm_review.zip`。包内包含 E3 scale 扫描、E3 eval、scale 复现检查配置、分析说明和 B 的指标汇总。其 `configs/e3/selection_lcm_lora.json` 使用未融合 LoRA，不能替换 E1 的默认融合配置。
- 压缩包汇总中的 24 行 E2 指标（12 个配置各 CLIP/ImageReward）与 GitHub E2 选择记录中的值和 500 样本数逐项一致。原包 SHA-256 见 provenance。
- B 的 E3 指标不混入 C 自己的实测 CSV，也不算作 C 的实验贡献。
