# 配置文件说明

每个 YAML 描述一组实验；`src.generate`、`src.evaluate`、`src.timing` 都只读这里的参数。

| 字段 | 含义 |
| --- | --- |
| `experiment` | 实验编号，如 `e1`、`e2`、`smoke`，决定输出目录 |
| `split` | `tune` / `eval` / `diversity`，对应 `data/splits/<split>.jsonl`；也可用 `prompts_file` 指定其他 prompt 文件。**必填**：用 `prompts_file` 时也要写，它决定输出目录（如 E5 可写 `split: parti`） |
| `prompts_file` | 自定义 prompt 文件（JSONL，每行至少有 `prompt_id` 和 `prompt` 两个字段），E5 等非 COCO 实验使用 |
| `limit` | 只取前 N 条 prompt（调试用） |
| `seeds` | seed 列表，每条 prompt 对每个 seed 各生成一张 |
| `method` | `sdxl_base` / `lcm_lora` / `sdxl_turbo` / `sdxl_lightning` |
| `resolution` | 生成分辨率（Turbo 512，其余 1024） |
| `dtype` | 计算精度，默认 `fp16` |
| `models` | `{仓库名: revision}`，已固定为冒烟测试实际使用的版本 |
| `timestep_spacing` | 仅基线：调度器的时间步间隔，默认 `trailing`（第一步从 t=999 开始） |
| `lightning_steps` | Lightning 使用哪个步数的 checkpoint（1 / 2 / 4 / 8） |
| `lora_fuse`、`lora_scale` | LCM-LoRA 是否融合及 adapter scale；E3 用 `lora_fuse: false` |
| `exploratory` | 设为 `true` 才允许超出官方推荐范围的设置（例如 Turbo、Lightning 开 CFG，或 Lightning 1 步）。它作用于整个文件，所以探索性的配置要单独放一个文件，不能和正式配置写在同一个网格里；这类文件不参与配置选择 |
| `name` | 单个配置的输出子目录名；不写时按参数自动生成。使用 `grid` 时总是自动生成，并包含每个被扫描的字段 |
| `save_size` | 保存尺寸，默认 512（指标都在 512 下计算，节省磁盘配额） |
| `full_res_first_n` | 前 N 条 prompt 另存原尺寸图到 `full/`，默认 20，用于局部裁剪图；图片本身已是保存尺寸时（如 Turbo 512）不另存 |
| `params` | 直接传给 pipeline 的参数，如 `num_inference_steps`、`guidance_scale`、`strength` |
| `grid` | 参数网格，每个组合生成一个配置点 |

## 网格扫描

`grid` 里的字段会自动放到正确的位置：上表中的配置字段（如 `lightning_steps`、`lora_scale`、`resolution`、`timestep_spacing`、`dtype`）覆盖顶层值，其余字段进入 `params`。例如 E3 的 adapter scale 扫描：

```yaml
method: lcm_lora
lora_fuse: false
lora_scale: 1.0
params:
  num_inference_steps: 4
  guidance_scale: 0.0
grid:
  lora_scale: [0.25, 0.5, 0.75, 1.0]
```

Lightning 的 `num_inference_steps` 由代码自动设为 `lightning_steps`，保证步数与 checkpoint 一致，不要在 `params` 里另写。Turbo 和 Lightning 没写 `guidance_scale` 时自动设为 0。

## 自动检查

- 顶层出现上表以外的字段（例如拼错的字段名）会报错。
- `params` 里出现上表中的配置字段会报错；`params` 里有 pipeline 不接受的参数名，会在加载模型后、生成第一张图之前报错。
- 同一个网格里两个配置点得到相同目录名会报错。
- guidance 超出方法的有效范围（Turbo、Lightning 必须为 0，LCM-LoRA 不超过 2.0）会报错，除非设置 `exploratory: true`。Lightning 的 `params` 里写了与 `lightning_steps` 不同的步数会报错；`lightning_steps: 1` 必须配合 `exploratory: true`；`timestep_spacing` 只能用于 `sdxl_base`。
- 生成和打分开始前，会先检查所有输出目录：已有的 `config.yaml` 与当前设置不一致时直接报错，不占用 GPU 时间，也不会把不同设置的图片混在一起打分。需要重跑时换一个名字，或把旧目录移走。
- 打分时如果图片数量不全会报错，确实需要时加 `--allow-partial`。
- `--only` 指定的名字不存在时报错。

输出位置：`outputs/<experiment>/<split>/<method>/<name>/`，内含图片 `<prompt_id>_<seed>.png`、`config.yaml`、每个分片一份 `run_info_<分片>.json`（库版本、GPU、模型 revision、调度器设置）和 UNet 调用日志 `calls_<分片>.jsonl`。
