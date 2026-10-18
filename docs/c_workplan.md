# C 的任务与交付清单

依据：`docs/plan.md`。`最新的分工.pdf` 与旧 PDF 的 SHA-256 相同；仓库计划补充了 trailing 基线、保存尺寸、配置冻结等细节。

## 范围

1. Turbo 的 E2：500 条 tune prompt，seed 0，512px，CFG 0，步数 1/2/3/4。
2. Lightning 的 E2：同一调参集，1024px，CFG 0，2/4/8 步分别加载对应完整 UNet。1 步 x0 模型单列探索，不参与主榜选择。
3. 调参集上按 ImageReward 排序、CLIP 不低于本方法最佳值 1.0 分的规则冻结质量点和最低步数点；交给 A/D。
4. 为完成交接，使用冻结的 Turbo/Lightning 配置生成 E1 图片、计算质量指标；C 作业固定在项目测速使用的 TC2N07，调用 A 的 `src.timing` 生成可交接的测速记录。A 仍负责跨方法的同硬件一致性核验和最终汇总。
5. 多样性：在 A 的固定 100 条 diversity prompt 上，每条生成 seed 0..7 的 8 张图片。每条 prompt 有 28 对，先平均 28 对 LPIPS，再对 100 条平均；不能把 2800 对当成 2800 个独立 prompt。
6. Methods、原论文证据、多样性分析、可视化、个人视频片段和贡献说明。

全部使用公开模型做推理与评估，不训练。E5 人工判定按 D 的组织安排参与。

## 运行方式

新增执行代码位于 `roles/c/`，调用 A 的 `src.generate`、`src.evaluate` 和 `src.select_configs`，不改公共实现。使用本角色分支 `c/adversarial-distillation`。

先创建仓库环境和数据：

```bash
bash slurm/setup_env.sh
mkdir -p logs
sbatch --job-name=c-data slurm/job.sbatch src.prepare_data
```

新账号若被默认 Anaconda channel 的 ToS 检查阻挡，使用 C 的安装入口 `bash slurm/c_setup_env.sh`，显式只用 conda-forge 创建 Python 环境，再安装 `environment.yml` 的原始 pip 锁定列表。它不接受默认频道条款，也不改全局频道设置。

按组长要求，先运行 `sbatch slurm/smoke.sbatch`，四个方法各生成 10 张并完成两种打分。随后用 `slurm/chains/c_smoke.txt` 额外检查探索性的 x0 scheduler（2 张）和环境；冒烟结果不能用于正式配置选择。

数据任务成功后，从登录节点提交两条 E2 链；每人最多两个作业：

```bash
sbatch --job-name=c-e2-turbo slurm/chain.sbatch slurm/chains/c_e2_turbo.txt 1 1
sbatch --job-name=c-e2-light slurm/chain.sbatch slurm/chains/c_e2_lightning.txt 1 2
```

按组长格式，C 的选择记录和合并网格写到 `configs/e1/selection_<方法>.json` 及 `configs/e1/<方法>.yaml`；另外在 `configs/c_frozen/` 保留逐点 E1、diversity YAML、数据划分 SHA-256 和执行清单。两种 E1 表达描述同一组冻结点，不重复生成图片。冻结发生在正式评测之前；重复执行不会覆盖不一致的冻结记录。不要提前手写“最佳配置”。

E2 成功后，运行对应后续任务（先确认原作业已结束）：

```bash
sbatch --job-name=c-final-turbo slurm/chain.sbatch slurm/chains/c_final_turbo.txt 1 2
sbatch --job-name=c-final-light slurm/chain.sbatch slurm/chains/c_final_lightning.txt 1 2
```

每个作业上限 6 小时。先根据冒烟测试和日志评估耗时，必要时按 `src.generate --shard` 拆分；超时后的生成可以续跑，未完成图片不能用来选配置。链脚本的阶段不会在计算节点上继续提交新作业。

## 四方法多样性的交接

预先确定只比较每个方法的 **质量优先点**；最低步数点属于 E1。组长明确要求基线使用 **50 步、CFG 7.5**，不使用最低步数点。Turbo/Lightning 会自动派生；基线从仓库已冻结记录派生：

```bash
python -m roles.c.workflow freeze --selection configs/e1/selection_sdxl_base.json
sbatch --job-name=c-div-base slurm/job.sbatch roles.c.workflow diversity --method sdxl_base
```

B 的 LCM 选择记录已从 `origin/b/e2-lcm-freeze` 读取并保存在 C 的输入目录，来源 commit 见 `configs/c_inputs/provenance.json`；不修改 B 的文件、不代表该分支已经合并。其质量点是 8 步、guidance 2.0、默认 scale 1.0。运行：

```bash
python -m roles.c.workflow freeze --selection configs/c_inputs/selection_lcm_lora.json
sbatch --job-name=c-div-lcm slurm/job.sbatch roles.c.workflow diversity --method lcm_lora
```

不使用 smoke 的 LCM 4 步配置冒充 B 的正式选定结果。可接收队友已经生成的同配置、同 prompt、同 seed 图片，以减少重复计算；先核对冻结记录、配置和 split hash。

## 结果文件

- 图片与每图分数：`outputs/<experiment>/<split>/<method>/<name>/`。
- 汇总 CSV：`outputs/metrics/metrics.csv`。
- 运行硬件和版本：每个图片目录的 `run_info_*.json`。
- 真实 UNet 步数：`calls_*.jsonl`。
- 用 `python -m roles.c.summarize` 导出 C 的指标、完整性检查和固定随机抽样网格。
- 报告方法/文献材料：`docs/c_methods_and_evidence.md`；视频提纲：`docs/c_video_outline.md`。

## 完成标准

不能仅凭代码准备好或 sbatch 接受就写“实验完成”。应核对：E2 正式 7 个点和探索 1 个点各 500 张及全量打分；两方法选择 JSON；冻结点 E1 每点 2000 张及全量指标；四方法多样性每点 800 张及 100 prompt 的 LPIPS；A 的统一延迟；可复现运行记录；交付图表和最终分析。缺 B/A 的交付时必须明确标记，不能补造结果。

当前文件是工作流程，不是实验完成证明。
