# 实验运行指南

三个入口：

| 脚本 | 用途 |
| --- | --- |
| `scripts/run_experiments.sh` | 方法 × 数据集 × repeat 正式评测 |
| `scripts/run_aflow.sh` | 在 validation 搜索 AFlow workflow，或用冻结 workflow 测试 |
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
DEEPSEEK_API_KEY=你的_API_KEY
GEMINI_API_KEY=你的_API_KEY
```

只需填写本次所选 profile 对应的 key。

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

相同实验设置再次运行时，脚本根据 `records.jsonl` 跳过已完成样本。遇到样本
error 时立即停止，错误诊断写入 `errors.jsonl`，但失败样本不写入
`records.jsonl`。人工修复后执行相同命令，会从该样本继续。

正式实验按模型、推理参数、method 设置、dataset/split、答案协议、环境、相关数据
文件及官方 checker 源码判断是否续跑；µ-MATH 还固定 judge profile 和评分协议。
runner 提交版本仍写入 manifest 留档，但无关代码提交、文档修改、增加其他模型配置
或数据仓库中无关文件的更新不会切断已有进度。确认页的 `Max new samples` 已扣除每个
cell 中完成的样本。

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
  --methods direct,pal,self_refine \
  --datasets gsm1k,math-perturb,harp,mathconstruct \
  --repeats 3 \
  --batch-size 10 \
  --concurrency 4 \
  --dry-run
```

分批正式运行：

```bash
./scripts/run_experiments.sh \
  --methods pal,self_refine \
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
direct, zero_shot_cot, pal, self_refine, all
```

AFlow 不属于该矩阵，统一通过下一节的专用脚本运行。

数据集：

| 名称 | test 数量 | 评分器 |
| --- | ---: | --- |
| `gsm1k` | 1,205 | 数值答案 |
| `math-perturb` | 230 | 官方 `answer_check` |
| `harp` | 4,302 | 官方 `latex_answer_check` |
| `harp-small` | 1,434 | HARP test 的固定分层 1/3 子集；官方 `latex_answer_check` |
| `u-math-text-only` | 720 | 锁定的 LLM judge |
| `mathconstruct` | 439 | 官方 `parse_and_check` |

U-MATH judge 已固定为 `qwen3.7-flash-2026-07-15`，使用
`configs/judges/u_math.lock.toml` 指向的固定 profile。它不接受命令行覆盖，也不
继承被测模型的 temperature、seed 或 repeat 设置；`--datasets all` 可以直接包含
U-MATH。
`--datasets all` 只包含五个完整基准，不包含 `harp-small`，避免重复评测 HARP
子集。需要小集时显式传入 `--datasets harp-small`；HARP validation 仅供优化与
选型，不是正式评测 dataset，不能通过该参数选择。

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
```

`--method-param` 可重复，且在 `--method-config` 之后应用。方法参数进入实验指纹。

### 产物

```text
results/experiments/
└── <model>/
    └── <method>/
        ├── <dataset-a>__r001__<tag>__<UTC-time>__<fingerprint>/
        ├── <dataset-a>__r002__<tag>__<UTC-time>__<fingerprint>/
        └── <dataset-b>__r001__<tag>__<UTC-time>__<fingerprint>/
```

每个实验单元目录内包含 `experiment.json`、`records.jsonl`、
`api_calls.jsonl`、`errors.jsonl`（发生错误时）和 `summary.json`。
`method_failed` 表示方法已完成模型调用但未产生可评分结果，按错误答案计入准确率并继续；
API、执行环境或评分器错误不写入当前样本，运行立即停止，修复后可续跑。

## AFlow：`run_aflow.sh`

脚本一次只处理一个数据集，支持 `gsm1k`、`math-perturb`、`harp` 和
`u-math-text-only`。不接受 MathConstruct，也不包含其他方法的矩阵参数。

### 搜索模式

搜索模型迭代提出 declarative workflow，执行模型只在对应的固化 validation 上评估；
每轮保存样本级结果，最终按 validation accuracy 选择 workflow，同分时优先模型调用
更少、节点更少的图。

```bash
./scripts/run_aflow.sh \
  --mode search \
  --dataset gsm1k \
  --model qwen35_flash \
  --optimizer-model qwen35_flash \
  --search-rounds 3 \
  --validation-size all \
  --concurrency 4
```

调试时可用 `--validation-size 10`。默认 optimizer 使用 `--model`；
`--optimizer-temperature`、`--optimizer-max-output-tokens` 和
`--optimizer-seed` 只控制优化器调用。搜索产物位于：

```text
results/aflow/<execution-model>/aflow/
└── search__<dataset>__<UTC-time>__<fingerprint>/
    ├── search.json
    ├── search_state.json
    ├── optimizer_api_calls.jsonl
    ├── candidates/round-*/
    └── workflow.json
```

### 测试模式

`--workflow` 必须指向搜索生成的 `workflow.json` 或其目录。脚本检查 workflow 已冻结、
没有使用 test 优化，并且绑定的数据集与 `--dataset` 完全一致。

```bash
./scripts/run_aflow.sh \
  --mode test \
  --dataset gsm1k \
  --model qwen35_flash \
  --workflow results/aflow/<execution-model>/aflow/<search-id>/workflow.json \
  --batch-size 100 \
  --repeats 3 \
  --seed 42 \
  --seed-mode increment \
  --concurrency 4
```

再次执行同一设置会按样本续跑。测试结果位于
`results/aflow/<execution-model>/aflow/<dataset>__rNNN__.../`，与对应模型的搜索目录同层。`--batch-size`、
`--repeats` 和 `--seed-mode` 只用于 test；`--search-rounds` 和
`--validation-size` 只用于 search。

`UTC-time` 格式为 `YYYYMMDDTHHMMSSZ`，记录该指纹首次创建时间；续跑复用原目录。

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
  --judge qwen37_flash \
  --batch-size 10 \
  --concurrency 4 \
  --dry-run
```

先运行一条：

```bash
./scripts/run_mu_math.sh --judge qwen37_flash --batch-size 1
```

分批运行或完成剩余数据：

```bash
./scripts/run_mu_math.sh \
  --judge qwen37_flash \
  --batch-size 100 \
  --concurrency 8

./scripts/run_mu_math.sh \
  --judge qwen37_flash \
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

内置候选：

| `--judge` | model | API key 环境变量 |
| --- | --- | --- |
| `qwen35_flash`（默认） | `qwen3.5-flash-2026-02-23` | `DASHSCOPE_API_KEY_BEIJING` |
| `qwen37_flash` | `qwen3.7-flash-2026-07-15` | `DASHSCOPE_API_KEY_BEIJING` |
| `gemini36_flash` | `gemini-3.6-flash`，reasoning effort=`medium` | `GEMINI_API_KEY` |
| `deepseek_v4_pro` | `deepseek-v4-pro`，thinking=`enabled`、reasoning effort=`high` | `DEEPSEEK_API_KEY` |
| `deepseek_v4_flash` | `deepseek-v4-flash`，thinking=`enabled`、reasoning effort=`high` | `DEEPSEEK_API_KEY` |

Qwen 候选固定 `temperature=0`、`top_p=1`、`seed=20260729`；Gemini 3.6
Flash 免费层使用 Google AI Studio API key，Gemini 与 DeepSeek thinking 模式均不
发送 `temperature`、`top_p` 或 `seed`。所有候选均固定
`max_output_tokens=4096`。

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
└── <judge-profile>/
    └── judge/
        └── official-test__<UTC-time>__<fingerprint>/
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
./scripts/run_aflow.sh --help
./scripts/run_mu_math.sh --help
```
