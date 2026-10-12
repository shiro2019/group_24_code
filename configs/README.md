# 配置文件说明

每个 YAML 描述一组实验；`src.generate`、`src.evaluate`、`src.timing` 都只读这里的参数。

| 字段 | 含义 |
| --- | --- |
| `experiment` | 实验编号，如 `e1`、`e2`、`smoke`，决定输出目录 |
| `split` | `tune` / `eval` / `diversity`，对应 `data/splits/<split>.jsonl`；也可用 `prompts_file` 指定其他 prompt 文件 |
| `limit` | 只取前 N 条 prompt（调试用） |
| `seeds` | seed 列表，每条 prompt 对每个 seed 各生成一张 |
| `method` | `sdxl_base` / `lcm_lora` / `sdxl_turbo` / `sdxl_lightning` |
| `resolution` | 生成分辨率（Turbo 512，其余 1024） |
| `models` | `{仓库名: revision}`，冒烟测试后固定，留空表示用最新版本 |
| `lightning_steps` | Lightning 使用哪个步数的 checkpoint（1 / 2 / 4 / 8） |
| `lora_fuse`、`lora_scale` | LCM-LoRA 是否融合及 adapter scale；E3 用 `lora_fuse: false` |
| `name` | 配置名，决定输出子目录；使用 `grid` 时自动生成 |
| `params` | 直接传给 pipeline 的参数，如 `num_inference_steps`、`guidance_scale` |
| `grid` | 参数网格，每个组合生成一个配置点 |
| `pipeline_keys` | 列在这里的 `grid` 字段会覆盖顶层字段（如 `lightning_steps`、`lora_scale`），而不是 `params` |

Lightning 扫描不同 checkpoint 的写法示例：

```yaml
lightning_steps: 4
pipeline_keys: [lightning_steps]
grid:
  lightning_steps: [2, 4, 8]
```

Lightning 的 `num_inference_steps` 由代码自动设为 `lightning_steps`，保证步数与 checkpoint 一致，不需要也不应在 `params` 里另写。

输出位置：`outputs/<experiment>/<split>/<method>/<name>/`，内含图片 `<prompt_id>_<seed>.png`、`config.yaml`、`run_info.json` 和 UNet 调用日志。
