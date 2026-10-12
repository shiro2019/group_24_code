# 项目方案（Group 24）

本文件是组内统一的实验方案和工作约定。动手写代码或跑实验前先读完；方案有改动时直接改这里，并在 PR 里说明。

## 1. 题目与研究问题

**题目**：Quality, Speed and Diversity in Few-Step Text-to-Image Generation: An Empirical Comparison of Distilled SDXL Models

- **RQ1**：在同一套评测协议下，三种少步数方法与多步 SDXL 基线相比，在图像质量、文本对齐、多样性和端到端延迟上分别处于什么位置？
- **RQ2**：在各方法的有效配置范围内，我们能控制的因素（基线步数、LCM-LoRA 步数和 adapter scale、Turbo 步数）如何影响这些指标？
- **RQ3**：Turbo 草图加 SDXL 精修的两阶段组合，在相近延迟预算下能否优于直接使用单一少步数方法？

只用公开预训练模型做推理和评测，不从头训练。所有数值由我们自己跑出，论文数值只作背景引用。

## 2. 方法与有效配置

每个方法只在官方说明的有效范围内运行；超出范围的设置标为探索性，不进主榜。

| 方法 | 加载方式 | Scheduler | 步数 | Guidance | 原生分辨率 | 注意事项 |
| --- | --- | --- | --- | --- | --- | --- |
| SDXL + DPM-Solver++（基线） | `stabilityai/stable-diffusion-xl-base-1.0` | `DPMSolverMultistepScheduler` | 4 / 8 / 15 / 25 / 50 | CFG 5.0 / 7.5，在调参集上选定 | 1024 | 全部是我们的设计选择，不是官方约束 |
| LCM-LoRA | base + `latent-consistency/lcm-lora-sdxl`；主实验 `fuse_lora`，E3 不融合、用 `set_adapters` 设 scale | `LCMScheduler` | 2 / 4 / 6 / 8 | 关闭，或 1.5 / 2.0（官方范围 1.0–2.0） | 1024 | 融合后再改调用参数里的 scale 不起作用；E1 固定用默认 scale 1.0 |
| SDXL-Turbo | `stabilityai/sdxl-turbo` | 管线默认 | 1 / 2 / 3 / 4 | 0.0（不用 CFG 和 negative prompt） | 512 | img2img 实际执行 `int(步数 × strength)` 步，须 ≥ 1 |
| SDXL-Lightning | base + `ByteDance/SDXL-Lightning` 的 `sdxl_lightning_{2,4,8}step_unet` | `EulerDiscreteScheduler`，`timestep_spacing="trailing"` | 2 / 4 / 8，每个步数一个单独 checkpoint | 0 | 1024 | 不同步数是不同模型，不能当作同一模型的步数消融；1 步 checkpoint 需 `prediction_type="sample"`，只作探索 |

来源：SDXL-Turbo、SDXL-Lightning、LCM-LoRA SDXL 的 Hugging Face 模型卡。具体 checkpoint revision 在 `configs/` 中固定。

**公平性与分辨率**：四个方法的训练过程、guidance 和 scheduler 都不同，对比只能说明“按推荐配置部署时”的表现，不能归因到某个训练组件。结果分两张表报告：

- **主榜（按推荐配置部署）**：各方法用推荐配置和原生分辨率生成，统一缩放到 512×512 后算指标。主表同时列出生成分辨率和指标分辨率。
- **受控对比（1024）**：基线、LCM-LoRA、Lightning 统一在 1024 下比较；Turbo 在 1024 下的结果单列为探索性参考。

## 3. 实验

共五组实验。所有实验固定 prompt 集合、seed 列表、fp16 精度和评测脚本版本。生成可以在不同设备上进行，但要记录硬件和库版本；延迟和显存统一由 A 在同一种 GPU 上重测。

| 编号 | 回答 | 数据阶段 | 改变的变量 | 固定的变量 | 指标 | 能支持的结论 |
| --- | --- | --- | --- | --- | --- | --- |
| E1 主对比 | RQ1 | 评测集 | 方法（按选择规则冻结的质量优先点和最低步数点） | 评测集 prompt、seed | FID、CLIP Score、ImageReward、端到端延迟、显存；多样性单独成表 | 各方法按推荐配置部署时的相对位置；不能归因到训练机制 |
| E2 方法内扫描 | RQ2 | 调参集 | 见第 2 节各方法的步数和 guidance；Lightning 标注为跨 checkpoint，1 步作探索 | 方法本身、调参集 prompt、seed | CLIP Score、ImageReward、延迟（不算 FID） | 每个方法的质量–速度曲线，并选出配置 |
| E3 LCM-LoRA adapter scale | RQ2 | 调参集扫描；代表配置在评测集复测 | scale 0.25 / 0.5 / 0.75 / 1.0（不融合） | 步数、guidance、prompt、seed | 调参：CLIP Score、ImageReward；代表配置补算 FID | adapter 强度对少步数生成的影响（组件敏感性实验） |
| E4 Hybrid refine | RQ3 | 调参集选 strength；评测集做最终比较 | 精修有无、strength | Turbo 草图、放大方法、两个阶段的 seed | 调参：CLIP Score、ImageReward、延迟；最终：同 E1，加原尺寸局部裁剪图 | 质量–延迟曲线上，两阶段组合是否优于单一方法 |
| E5 困难 prompt | RQ1 | PartiPrompts 子集 | prompt 类别 | E1 冻结配置、seed | 人工判定失败率；CLIP Score、ImageReward 分组；典型失败案例 | 各方法在数量、文字、空间关系上的实测失败率 |

### 配置选择规则（跑评测集前冻结）

1. **质量优先点**：在官方有效范围内，选调参集 ImageReward 最高的配置。
2. **CLIP 检查**：若该配置的 CLIP Score 比同方法最佳值低 1.0 分以上（0–100 分制），顺延到下一名。
3. **最低步数点**：取有效范围内的最少步数，其余参数与质量优先点相同；不声称它一定最快。
4. E1 中的 LCM 固定使用默认 adapter scale 1.0，E3 单独报告；调优后的 scale 若展示，标为“小组调优变体”。E4 的代表配置同样在调参集上选定后冻结。
5. 最终评测使用冻结的配置集合和 seed 列表，不根据评测集结果重新调参。

### E4 配置

| 项目 | 设置 |
| --- | --- |
| 第一阶段 | SDXL-Turbo，1 步，512×512，guidance 0，草图 seed 列表固定 |
| 放大 | 512→1024，bicubic 像素放大后 VAE 编码；“无精修”对照用同一张草图和同一放大方法 |
| 第二阶段 | SDXL base img2img，`DPMSolverMultistepScheduler`，名义步数 10，strength 0.2 / 0.3 / 0.4 / 0.5，CFG 与基线选定值相同，加噪 seed 列表单独固定 |
| 记录 | 实际执行的 timestep 数、UNet forward 调用次数和每次调用的 batch size，以日志为准；`int(名义步数 × strength)` 只作预期值。CFG 是把条件与无条件分支拼成一个 batch 计算，不等于调用次数翻倍；报告 NFE 时写明计数方式 |
| 对照组 | Turbo 4 步、Lightning 4 步、LCM 4 步、基线低步数，在同一种 GPU 上实测延迟 |
| 比较方式 | 画质量–延迟曲线，不强行寻找延迟完全相同的点；1024 细节用固定位置的原尺寸裁剪图展示 |

E4 是已有方法的组合探索，不写成原创方法，也不要求一定成功；如果成本增加却没有稳定改善，就分析原因写进报告。

### E5 规则

- 数据：`google-research/parti` 仓库的 `PartiPrompts.tsv`（共 1,632 条），取 Challenge 为 Quantity（90 条）、Writing & Symbols（91 条）、Properties & Positioning（35 条）三类。
- 每类用固定随机种子抽 30 条，不重复抽样。
- 生成之前人工确认每条 prompt 确实在测试该能力，不合适的替换并记录原因；名单冻结后再生成和判定，不根据生成结果调整样本。
- 判定规则：数量——指定对象数量是否正确；文字——要求的文本是否准确出现；空间关系——指定的相对位置是否成立。
- 判定时隐藏方法名；另一人独立复核 20% 的样本，不一致时讨论后定。
- 结论表述为“所选子集上的实测失败率”；只用自动指标的部分，表述为“自动指标差异与典型失败现象”。

### 证据边界

“对抗损失导致更锐利”“teacher CFG 的作用”这类训练机制的结论，只引用原论文的消融实验，并标明是文献证据。不预设任何模型存在 mode collapse，多样性结论只看数据。

## 4. 数据划分与评测协议

**数据划分**

- MS-COCO 2017 val：用固定随机种子抽取 image_id，每个 image_id 只取第 1 条 caption，规则写进代码。
- 按 image_id 划分：调参集 500 个、评测集 2,000 个，互不重叠。
- 所有方法共用同一份 seed 列表；多样性实验从评测集中固定取 100 个 prompt × 8 个 seed。
- COCO 可能在这些模型的预训练数据里，无法核实，因此不宣称“未见数据上的泛化”。

**指标**

| 阶段 | 生成 prompt | FID 真实参考集 | 计算的指标 |
| --- | --- | --- | --- |
| 参数搜索与配置选择（E2，E3、E4 的调参） | 调参集 | 不算 FID | CLIP Score、ImageReward、延迟 |
| 最终比较（E1，E3、E4 的代表配置） | 评测集 | 评测集对应的真实图像 | FID、CLIP Score、ImageReward、延迟、显存 |
| 多样性 | 评测集中固定的 100 个 prompt × 8 个 seed | 不适用 | 平均 LPIPS，单独成表 |
| 困难 prompt（E5） | PartiPrompts 子集 | 不适用 | 人工判定失败率、CLIP Score、ImageReward |

- FID 用 clean-fid，预处理统一；样本量有限会带来偏差，只用于组内相对比较，不和论文数值对照。
- CLIP Score、ImageReward 固定模型版本，写在代码里。
- LPIPS 高不等于更好，必须结合文本对齐和图片一起解释。
- 不靠单一指标判定胜负。

**计时**：所有方法在同一种 GPU、fp16、batch size 1 下测，先预热；计时从文本编码开始到 VAE 解码完成，前后各调用一次 `torch.cuda.synchronize()`；两阶段方法算全流程；报告中位数和四分位距。

**可视化**：对比网格图用固定随机抽样的 prompt，不挑图；失败案例单独展示，并标明是挑选出来的。

## 5. 分工

| 角色 | 职责 | 预估 GPU 用时 | 依赖 | 验收标准 | 报告章节 | 视频片段 |
| --- | --- | --- | --- | --- | --- | --- |
| A：基线与评估管线（兼协调人） | 环境、数据划分、生成与评估脚本；基线的 E1、E2；Turbo 1024 受控对比；统一计算全部指标和延迟 | 约 7–9 小时 | 无 | 10/13：四个方法各生成 10 张图，指标输出为 CSV；10/18：指标总表和延迟表 | 实验设置、评测协议、基线结果 | 任务与实验设置 |
| B：一致性蒸馏 | LCM-LoRA 原理；LCM 的 E2；E3 | 约 2–3 小时 | A 的脚本 | 10/15：调参集扫描完成并选定配置；10/17：E3 完成，并确认不同 scale 的输出确实不同、同一配置重复运行可复现 | Methods（LCM）、E3 | LCM 与组件实验 |
| C：对抗蒸馏 | Turbo、Lightning 原理，整理原论文消融作文献证据；Turbo、Lightning 的 E2（含 Lightning 1 步）；多样性实验 | 约 2–3 小时 | A 的脚本 | 10/15：调参集扫描完成并选定配置；10/17：多样性实验完成 | Methods（ADD、Lightning）、多样性分析 | 对抗蒸馏与多样性 |
| D：改进探索与整合 | E4 及对照组；E5 的组织与失败案例（人工判定四人分担）；统一绘图；合并报告、剪视频 | 约 1.5–2.5 小时 | A 的脚本；B、C 选定的配置 | 10/14：refine 跑通并记录实际步数；10/17：E4、E5 完成；10/21：视频成片 | Introduction、E4、E5、Conclusion | 改进探索与总结 |

GPU 用时是估算，10/13 冒烟测试后用实测速度更新。

## 6. 时间线

| 日期 | 内容 |
| --- | --- |
| 10/12–10/13 | 环境与评估脚本（A 在 10/13 交付） |
| 10/14–10/15 | 调参集扫描（E2） |
| 10/16–10/17 | 评测集 E1，及 E3–E5 |
| 10/18 | 冻结结果，之后不再新增实验 |
| 10/19–10/20 | 报告初稿与 slides |
| 10/21 | 各自录制，D 剪辑 |
| 10/22 23:59 | 提交 presentation 视频 |
| 10/23–10/24 | 根据反馈润色报告，提前一天提交三份文件（10/25 截止） |

## 7. 工作约定

**代码与目录**

- `src/` 放公共代码，由 A 维护：数据划分、生成、评估、计时。其他人在此基础上写自己的实验，不另起一套生成或评估代码；需要改公共代码时单独开 PR 说明原因。
- `configs/<实验>/<方法>.yaml` 放每个实验的全部参数（模型、checkpoint revision、scheduler、步数、guidance、分辨率、seed 列表、数据划分），代码里不手改参数。
- 生成结果放 `outputs/<实验>/<划分>/<方法>/<配置名>/`，图片命名 `<prompt_id>_<seed>.png`，同目录保存本次运行的配置和日志。
- 指标统一写成 CSV，列为：`experiment, split, method, config, metric, value, n_samples, hardware, timestamp`。D 用它统一画图。
- 以上接口是初始约定，A 在 10/13 交付脚本时可以调整，并同步更新本节。

**Git 流程**

- 不直接向 `main` 推送；在自己的分支开发，分支名用 `<角色>/<内容>`，例如 `b/lcm-sweep`。
- 完成后开 PR，至少一人审核后合并；PR 描述写清楚做了什么、跑了哪些配置，方便最后写 contribution。
- 只改自己负责的部分；改动别人的文件先在群里说。
- 不提交数据集、模型权重、生成图片和任何密钥。

**进度**：每天晚上在群里一句话同步进度和遇到的问题。
