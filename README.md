# group_24_code

CA6127 Transformer and Generative Adversarial Networks — Group 24 课程项目代码。

**题目**：Quality, Speed and Diversity in Few-Step Text-to-Image Generation: An Empirical Comparison of Distilled SDXL Models

比较 SDXL + DPM-Solver++（基线）、LCM-LoRA、SDXL-Turbo、SDXL-Lightning 在图像质量、文本对齐、多样性和端到端延迟上的取舍。实验设计、分工、时间线和工作约定见 [docs/plan.md](docs/plan.md)，开始工作前先读完。

## 目录结构

```
docs/      项目方案与工作约定
configs/   每个实验的配置文件（方法、步数、guidance、seed 等），不在代码里手改参数
src/       生成、评估、计时等 Python 代码
slurm/     集群作业脚本
data/      数据集与 prompt 划分（不提交到仓库）
outputs/   生成的图片、指标 CSV、日志（不提交到仓库）
```

## 环境

```bash
conda env create -f environment.yml
conda activate ca6127
```

库版本在 10/13 冒烟测试通过后由 A 固定到 `environment.yml`。

## 在集群上运行

```bash
git clone https://github.com/shiro2019/group_24_code.git && cd group_24_code
bash slurm/setup_env.sh                                  # 只需一次：创建 conda 环境和 logs/
sbatch --job-name=data slurm/job.sbatch src.prepare_data  # 只需一次：COCO 划分与 FID 参考图
sbatch slurm/smoke.sbatch                                 # 冒烟测试：四个方法各 10 张图并打分
```

常用命令（都在仓库根目录提交）：

```bash
sbatch --job-name=e2-base slurm/job.sbatch src.generate --config configs/e2/sdxl_base.yaml
sbatch --job-name=e2-base-shard0 slurm/job.sbatch src.generate --config configs/e2/sdxl_base.yaml --shard 0 --num-shards 2
sbatch --job-name=eval slurm/job.sbatch src.evaluate --config configs/e2/sdxl_base.yaml --metrics clip,imagereward
sbatch --job-name=timing --nodelist=TC2N07 slurm/job.sbatch src.timing --config configs/e1/sdxl_base.yaml
```

生成可中断重跑，已存在的图片会跳过；指标追加到 `outputs/metrics/metrics.csv`。配置字段见 [configs/README.md](configs/README.md)。

## 协作流程

1. 不直接向 `main` 推送；每人在自己的分支开发，分支名用 `<角色>/<内容>`，例如 `a/eval-pipeline`、`b/lcm-sweep`。
2. 完成后开 Pull Request，由 A 检查脚本接口后合并。
3. 实验参数写进 `configs/`，每次运行保存所用配置和日志。
4. 不提交数据集、模型权重、生成图片和任何密钥。

## 最终提交

截止前把代码打包为 `group_24_code.zip` 提交到 NTULearn，不含数据集、模型和外部链接；同时提交 `group_24_report.pdf` 和 `group_24_contribution.pdf`。
