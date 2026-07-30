# 实验运行指南

两个入口：

| 脚本 | 用途 |
| --- | --- |
| `scripts/run_experiments.sh` | 方法 × 数据集 × repeat 正式评测 |
| `scripts/run_mu_math.sh` | 在 µ-MATH 上评估候选 U-MATH judge |

以下命令均在 `benchmark-runner` 根目录执行。

## 环境配置

```bash
python3 -m venv .venv-checkers
.venv-checkers/bin/pip install -r requirements/experiment-lock.txt
.venv-checkers/bin/pip install -e . --no-deps
cp .env.example .env
```

在 `.env` 中填写：

```dotenv
DASHSCOPE_API_KEY_BEIJING=你的_API_KEY
```

运行前检查：

```bash
.venv-checkers/bin/pip check
.venv-checkers/bin/python scripts/verify_workspace.py
.venv-checkers/bin/python ../math-benchmark-data/scripts/validate_datasets.py
MATH_CHECKER_PYTHON="$PWD/.venv-checkers/bin/python" \
  .venv-checkers/bin/python -m unittest discover -v
```

## 共享参数语义

### 分批与续跑

`--batch-size N` 每次处理接下来的 N 条未完成样本；`all` 处理全部剩余样本。

```bash
--batch-size 100
--batch-size all
```

相同实验设置再次运行时，脚本根据 `records.jsonl` 跳过已完成样本。已记录为 error
的样本也视为完成。

### 并发

`--concurrency N` 设置单个实验单元内的并发样本数，默认 `1`。

- 最多只有 N 条样本在途；
- 主线程按数据集顺序写入结果；
- `batch-size` 和 `concurrency` 均不进入实验指纹，续跑时可以调整；
- 通用矩阵的不同 method/dataset/repeat 单元仍顺序执行。

profile 中的 `min_request_interval_seconds` 限制同一 backend 的请求启动频率。默认
`1.0` 表示请求启动至少间隔一秒；提高并发只会重叠响应等待。API 配额允许时可降低：

```toml
min_request_interval_seconds = 0.2
```

### 实验指纹

模型、推理参数、方法参数、数据集、repeat、run tag、judge、代码/数据 revision 和
Python 环境进入实验指纹。改变这些设置会创建新目录，不会读取旧实验。

`batch-size`、`concurrency`、`yes`、`debug` 和 `skip-preflight` 不进入指纹。

## 通用评测：`run_experiments.sh`

### 示例

默认运行 `direct × gsm1k × repeat 1` 的下一条样本：

```bash
./scripts/run_experiments.sh
```

查看计划，不调用 API：

```bash
./scripts/run_experiments.sh \
  --methods direct,pal,self_refine,aflow \
  --datasets gsm1k,math-perturb,harp,mathconstruct \
  --repeats 3 \
  --batch-size 10 \
  --concurrency 4 \
  --dry-run
```

分批正式运行：

```bash
./scripts/run_experiments.sh \
  --methods pal,self_refine,aflow \
  --datasets gsm1k,math-perturb,harp,mathconstruct \
  --repeats 3 \
  --batch-size 100 \
  --concurrency 8
```

完成单个实验的全部剩余样本：

```bash
./scripts/run_experiments.sh \
  --methods pal \
  --datasets gsm1k \
  --batch-size all \
  --concurrency 8
```

### 方法与数据集

方法：

```text
direct, zero_shot_cot, pal, self_refine, aflow, all
```

当前 AFlow 只评估固化 workflow，不执行在线 optimizer。

数据集：

| 名称 | test 数量 | 评分器 |
| --- | ---: | --- |
| `gsm1k` | 1,205 | 数值答案 |
| `math-perturb` | 230 | 官方 `answer_check` |
| `harp` | 4,302 | 官方 `latex_answer_check` |
| `u-math-text-only` | 720 | 锁定的 LLM judge |
| `mathconstruct` | 439 | 官方 `parse_and_check` |

当前 U-MATH judge lock 为 `pending`，所以选择 `u-math-text-only` 或
`--datasets all` 会在创建产物和调用 API 前退出。

### 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--model MODEL` | `qwen35_flash` | profile 名称、已配置 model ID 或 TOML 路径 |
| `--model-id ID` | profile 值 | 覆盖实际 model ID |
| `--methods LIST` | `direct` | 逗号分隔的方法或 `all` |
| `--datasets LIST` | `gsm1k` | 逗号分隔的数据集或 `all` |
| `--repeats N` | `1` | 每个 method × dataset 的重复次数 |
| `--batch-size N\|all` | `1` | 每个实验单元本次新增样本数 |
| `--concurrency N` | `1` | 每个实验单元内并发数 |
| `--temperature X` | profile 值 | 范围 `0`–`2` |
| `--top-p X` | profile 值 | 范围 `0`–`1` |
| `--max-output-tokens N` | profile 值 | 单次模型调用最大输出 |
| `--seed N` | profile 值或不发送 | 范围 `0`–`2147483647` |
| `--seed-mode MODE` | `fixed` | `fixed` 或 `increment` |
| `--method-param SPEC` | 无 | `METHOD.KEY=JSON_VALUE` |
| `--method-config SPEC` | 无 | `METHOD=/path/config.json` 或 TOML |
| `--run-tag TAG` | 空 | 新实验的可读标签，进入指纹 |
| `--data-root PATH` | `../math-benchmark-data` | 数据仓库 |
| `--output-root PATH` | `results/experiments` | 结果目录 |
| `--checker-python PATH` | `.venv-checkers/bin/python` | 官方 checker 解释器 |
| `--env-file PATH` | `.env` | 环境变量文件 |
| `--dry-run` | 关闭 | 只检查和展示计划 |
| `--yes` | 关闭 | 跳过确认 |
| `--skip-preflight` | 关闭 | 跳过 API 连通性检查 |
| `--debug` | 关闭 | 在终端打印完整 traceback |

### Seed

```bash
./scripts/run_experiments.sh \
  --methods direct \
  --datasets gsm1k \
  --repeats 3 \
  --seed 1234 \
  --seed-mode increment \
  --batch-size 100
```

- `fixed`：所有 repeat 使用同一 seed；
- `increment`：第 n 个 repeat 使用 `seed + n - 1`，必须提供 base seed。

seed 不能保证远端服务完全确定。`temperature=0` 且服务确定时，相同 seed 的多个
repeat 可能完全一致。

### 方法参数

```bash
./scripts/run_experiments.sh \
  --methods self_refine \
  --datasets gsm1k \
  --method-param self_refine.max_refinements=2

./scripts/run_experiments.sh \
  --methods aflow \
  --datasets gsm1k \
  --method-config aflow=/absolute/path/to/override.toml
```

`--method-param` 可重复，且在 `--method-config` 之后应用。方法参数进入实验指纹。

### 产物

```text
results/experiments/
└── <model>__<method>__<dataset>__r001__<tag>__<fingerprint>/
    ├── experiment.json
    ├── records.jsonl
    ├── api_calls.jsonl
    ├── errors.jsonl
    └── summary.json
```

| 文件 | 内容 |
| --- | --- |
| `experiment.json` | 完整配置、环境、revision 和指纹 |
| `records.jsonl` | 题目、参考答案、metadata、输出、评分和耗时 |
| `api_calls.jsonl` | API latency、token、retry、seed 和 request ID |
| `errors.jsonl` | 样本错误和 traceback；有错误时生成 |
| `summary.json` | 累计指标及本次 batch/concurrency |

## µ-MATH judge：`run_mu_math.sh`

官方 test 包含 271 道题 × 4 个候选答案来源模型，共 1,084 行。该脚本只评估候选
judge，不运行论文方法，也不读取正式 U-MATH judge lock。

### 示例

检查计划：

```bash
./scripts/run_mu_math.sh \
  --judge qwen35_flash \
  --batch-size 10 \
  --concurrency 4 \
  --dry-run
```

先运行一条：

```bash
./scripts/run_mu_math.sh --judge qwen35_flash --batch-size 1
```

分批运行或完成剩余数据：

```bash
./scripts/run_mu_math.sh \
  --judge qwen35_flash \
  --batch-size 100 \
  --concurrency 8

./scripts/run_mu_math.sh \
  --judge qwen35_flash \
  --batch-size all \
  --concurrency 8
```

### 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--judge JUDGE` | `qwen35_flash` | 候选 profile 名称、已配置 model ID 或 TOML 路径 |
| `--batch-size N\|all` | `1` | 本次新增行数 |
| `--concurrency N` | `1` | judge 请求并发数 |
| `--data-root PATH` | `../math-benchmark-data` | 数据仓库 |
| `--output-root PATH` | `results/mu_math` | 结果目录 |
| `--env-file PATH` | `.env` | 环境变量文件 |
| `--dry-run` | 关闭 | 只检查和展示计划 |
| `--yes` | 关闭 | 跳过确认 |
| `--skip-preflight` | 关闭 | 跳过 judge API 连通性检查 |
| `--debug` | 关闭 | 在终端打印完整 traceback |

该脚本不提供 model ID、temperature、seed 或 max token 的命令行覆盖。候选 judge
配置完全由 profile 固定。

### Judge profile

默认文件：`configs/judges/candidates/qwen35_flash.toml`。

```toml
profile = "candidate-judge-qwen35-flash-bj"
provider = "qwen"
api_type = "openai"
model = "qwen3.5-flash-2026-02-23"
base_url = "https://example.com/compatible-mode/v1"
api_key_env = "DASHSCOPE_API_KEY_BEIJING"
enable_thinking = false
temperature = 0.0
top_p = 1.0
max_output_tokens = 4096
seed = 20260729
request_timeout_seconds = 600
max_retries = 5
min_request_interval_seconds = 1.0
```

修改 profile 会创建新实验目录。测试其他 judge 时建议复制为新文件：

```bash
cp configs/judges/candidates/qwen35_flash.toml \
  configs/judges/candidates/my_candidate.toml

./scripts/run_mu_math.sh \
  --judge configs/judges/candidates/my_candidate.toml \
  --batch-size 100
```

### 指标与产物

judge 输出 `Yes`、`No` 或 `Inconclusive`。格式不合规时按 `Inconclusive` 处理；
`Inconclusive` 始终算错，不映射成 Yes 或 No。

`summary.json` 报告总体和四个候选答案来源模型切片的：

- macro-F1、positive/negative F1、accuracy；
- TPR、TNR、PPV、NPV；
- TP、TN、FP、FN 和 Inconclusive；
- token、latency、retry 和错误。

```text
results/mu_math/
└── <judge-profile>__mu-math__<fingerprint>/
    ├── experiment.json
    ├── records.jsonl
    ├── api_calls.jsonl
    ├── errors.jsonl
    └── summary.json
```

## 错误与退出码

样本错误写入 `errors.jsonl`；`--debug` 同时在终端显示 traceback。未捕获异常写入
输出根目录的 `_fatal_errors.jsonl`。

| 退出码 | 含义 |
| ---: | --- |
| `0` | 成功、dry-run 或实验已完成 |
| `1` | 存在样本级 error |
| `2` | 参数、配置、仓库、数据、环境或 preflight 失败 |
| `130` | 用户中断；已落盘样本可续跑 |

```bash
./scripts/run_experiments.sh --help
./scripts/run_mu_math.sh --help
```
