# group_24_code

CA6127 Transformer and Generative Adversarial Networks — Group 24 课程项目代码。

**题目**：Quality, Speed and Diversity in Few-Step Text-to-Image Generation: An Empirical Comparison of Distilled SDXL Models

比较 SDXL + DPM-Solver++（基线）、LCM-LoRA、SDXL-Turbo、SDXL-Lightning 在图像质量、文本对齐、多样性和端到端延迟上的取舍。实验设计、分工、时间线和工作约定见 [docs/plan.md](docs/plan.md)，开始工作前先读完。

## 目录结构

```
docs/      项目方案与工作约定
configs/   每个实验的配置文件（方法、步数、guidance、seed 等），不在代码里手改参数
src/       生成、评估、计时等 Python 代码
roles/     基于 src/ 的实验编排脚本（roles/c：Turbo、Lightning 的调参、冻结、E1 和四方法多样性）
slurm/     集群作业脚本
data/      数据集与 prompt 划分（不提交到仓库）
outputs/   生成的图片、指标 CSV、日志（不提交到仓库）
```

## 环境

```bash
conda env create -f environment.yml
conda activate ca6127
```

`environment.yml` 中的版本已在 TC2 集群上验证。torch 必须用 CUDA 12.8 版本（集群驱动不支持 CUDA 13），transformers 必须用 4.x（ImageReward 不兼容 5.x），不要单独升级这两个包。

## 在集群上运行

```bash
git clone https://github.com/shiro2019/group_24_code.git && cd group_24_code
bash slurm/setup_env.sh                                  # 只需一次：创建 conda 环境和 logs/
sbatch --job-name=data slurm/job.sbatch src.prepare_data  # 只需一次：COCO 划分与 FID 参考图
sbatch slurm/smoke.sbatch                                 # 冒烟测试：四个方法各 10 张图并打分
```

常用命令（都在仓库根目录提交）：

```bash
# 短任务（6 小时内）：单个作业
sbatch --job-name=eval slurm/job.sbatch src.evaluate --config configs/e2/sdxl_base.yaml --metrics clip,imagereward
sbatch --job-name=timing --nodelist=TC2N07 slurm/job.sbatch src.timing --config configs/e1/sdxl_base.yaml

# 长任务：写成步骤清单，在一个作业里依次运行其中几步（例如 E2 基线实测共约 6.7 小时：3 个分片再打分）
J1=$(sbatch --parsable --job-name=e2-base slurm/chain.sbatch slurm/chains/e2_sdxl_base.txt 1 2)
sbatch --dependency=afterok:$J1 --kill-on-invalid-dep=yes --job-name=e2-base slurm/chain.sbatch slurm/chains/e2_sdxl_base.txt 3 4

# E4 两阶段（Turbo 草图 + SDXL 精修）：生成和计时都用 src.hybrid，打分仍用 src.evaluate
sbatch --job-name=e4-smoke slurm/job.sbatch src.hybrid --config configs/smoke/hybrid.yaml
sbatch --job-name=e4-timing --nodelist=TC2N07 slurm/job.sbatch src.hybrid --config configs/e4/hybrid.yaml --timing

# 调参集打分完成后，按 plan 的规则选出质量优先点和最低步数点（只读 CSV，不需要 GPU）
python -m src.select_configs --config configs/e2/sdxl_base.yaml
```

选择脚本只用于 E2 调参集（`split: tune`）的网格：只使用完整打分（图片数量齐全）的最新指标，结果连同全部候选配置的分数写到 `outputs/selection/<实验>_<方法>.json`，冻结后据此手动写 `configs/e1/<方法>.yaml`，并把这份 JSON 复制为 `configs/e1/selection_<方法>.json`，两者一起提交到仓库作为冻结记录（基线的例子见 `configs/e1/`）。标了 `exploratory: true` 的文件（如 Lightning 1 步）不参与选择。

集群每人最多同时提交 2 个作业、单个作业最长 6 小时，而且**计算节点上不能提交作业**，所以长任务要这样拆：在 `slurm/chains/` 里写一个步骤清单，每行一个 `<模块> <参数>`；`chain.sbatch <清单> <起始步> <结束步>` 在一个作业里依次运行这几步，每个作业的总时长控制在 6 小时以内。超过 6 小时就像上面那样分成两个作业，都在登录节点提交，第二个用 `--dependency=afterok` 排在第一个后面。步骤清单每行按空格切分参数，不支持引号，行尾也不能加注释。

某一步失败或作业超时，后面排队的作业也会自动取消。修好后按原来的分段，从失败的那一步重新提交两个作业，并写明结束步，保证每个作业不超过 6 小时。例如第 2 步失败时，第一个作业提交 `chain.sbatch <清单> 2 2`，第二个仍是 `3 4`，用 `--dependency=afterok` 排在后面。生成会从中断处继续。

生成可中断重跑，已存在的图片会跳过；指标追加到 `outputs/metrics/metrics.csv`，同一配置重复打分会追加多行，画图时取 `timestamp` 最新的一行。这个文件在每个人自己的集群目录里，汇总时需要把各人的 CSV 合并。`hardware` 列记录的是生成图片的 GPU（分片用了不同 GPU 时用 `/` 连接），延迟只看 `src.timing` 的结果（生成日志里的耗时没有预热，仅供参考）。

**修改配置后重跑**：生成和打分前会检查输出目录里的 `config.yaml`，设置不一致就拒绝运行。例如模型版本固定、基线改用 trailing 之后，旧的 `outputs/smoke` 必须先移走（如移到 `outputs/_discarded/`）才能重跑冒烟测试。配置字段见 [configs/README.md](configs/README.md)。

## 协作流程

1. 不直接向 `main` 推送；每人在自己的分支开发，分支名用 `<角色>/<内容>`，例如 `a/eval-pipeline`、`b/lcm-sweep`。
2. 完成后开 Pull Request，由 A 检查脚本接口后合并。
3. 实验参数写进 `configs/`，每次运行保存所用配置和日志。
4. 不提交数据集、模型权重、生成图片和任何密钥。

## 最终提交

截止前把代码打包为 `group_24_code.zip` 提交到 NTULearn，不含数据集、模型和外部链接；同时提交 `group_24_report.pdf` 和 `group_24_contribution.pdf`。
