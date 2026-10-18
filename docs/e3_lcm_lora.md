# E3 LCM-LoRA adapter scale 实验记录

本实验考察固定推理设置下 adapter scale 对 LCM-LoRA 输出的影响。在 2000 个评测样本上，scale=1.0 获得更高的 ImageReward 和更低的 FID，scale=0.75 的 CLIP Score 略高。

## 实验设置

使用 SDXL base 和 LCM-LoRA SDXL，模型 revision 固定在配置文件中。调度器为 LCMScheduler；LoRA 不融合，通过 set_adapters 设置 adapter scale。

生成固定为 8 步、guidance_scale=2.0、seed=0、fp16、1024×1024。图像统一保存为 512×512 用于评测，每个配置前 20 条 prompt 另存原尺寸图。生成与指标评测均使用 NVIDIA A40。

数据采用公共脚本生成的 MS-COCO 2017 val 划分：调参集 500 条、评测集 2000 条，按 image_id 互不重叠。划分随机种子为 2026，每张图选择 annotation id 最小的 caption。每个划分内，各配置共用相同 prompt 和 seed。

调参阶段计算 CLIP Score 和 ImageReward；代表配置在评测集上计算 CLIP Score、ImageReward 和 FID。FID 使用评测集对应的 2000 张真实图像，经中心裁剪和缩放至 512×512 后作为参考集。

## 调参结果与代表配置

每个配置包含 500 个样本。

| Adapter scale | CLIP Score ↑ | ImageReward ↑ |
| --- | ---: | ---: |
| 0.25 | 25.1011 | -0.9608 |
| 0.50 | 26.4257 | 0.0323 |
| 0.75 | 26.5002 | 0.5157 |
| 1.00 | 26.3467 | 0.6255 |

scale=1.0 的 ImageReward 最高，其 CLIP Score 比最佳值低 0.1535，满足 1.0 分的容差，选为质量优先点。另保留 CLIP Score 最高的 scale=0.75，作为评测集上的第二个代表配置。

代表配置在评测集生成前确定，选择依据和全部候选分数保存在 selection_lcm_lora.json 中。该记录保留调参阶段的分数、时间和代码版本。

## 评测集结果

每个配置包含 2000 个样本。

| Adapter scale | CLIP Score ↑ | ImageReward ↑ | FID ↓ |
| --- | ---: | ---: | ---: |
| 0.75 | 26.6067 | 0.5027 | 57.2222 |
| 1.00 | 26.5577 | 0.6405 | 51.0024 |

在本次固定设置和评测子集上，scale=1.0 相比 scale=0.75，ImageReward 高 0.1378，FID 低 6.2198，CLIP Score 低 0.0490。两个代表点体现了文本对齐指标与其他质量指标之间的取舍。

这些结果对应 seed=0、8 步和 guidance_scale=2.0。FID 用于本组相同样本量和预处理协议下的相对比较。正式延迟与显存由 A 按统一协议测量并汇总。

## 输出差异与重复运行检查

- 对前 10 条调参 prompt 运行四个 scale，共得到 40 张图像；每条 prompt 对应的四个保存 PNG 的 SHA-256 均不同。
- scale=1.0 在 TC2N02 的 NVIDIA A40 上独立重复生成相同的 10 条 prompt，保存 PNG 的 SHA-256 匹配 10/10。
- 准备 FID 参考图前后，tune、eval、diversity 三个划分文件的 SHA-256 均一致。
- 两个评测配置各有 2000 张生成图，评测集真实参考图为 2000 张；正式评测作业正常结束，六项指标均使用完整样本。

## 配置与结果位置

- [调参扫描配置](../configs/e3/lcm_lora.yaml)
- [代表配置选择记录](../configs/e3/selection_lcm_lora.json)
- [评测集配置](../configs/e3/eval_lcm_lora.yaml)
- [scale 冒烟检查配置](../configs/smoke/lcm_lora_scale.yaml)
- [重复运行检查配置](../configs/smoke/lcm_lora_scale_repeat.yaml)

原始指标位于 `outputs/metrics/metrics.csv`。汇总副本为 `outputs/handoff/b_lcm_metrics.csv`，包含 E2 调参 24 行、E3 调参 8 行、E3 评测 6 行，共 38 行。

生成图、逐图分数和运行元数据保存在 `outputs/e3/` 下。数据、模型、图像和指标 CSV 按仓库约定保留在 Git 忽略目录中。
