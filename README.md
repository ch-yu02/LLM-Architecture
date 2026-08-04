# LLM Architecture Math Benchmark Runner

一个面向数学推理方法对比的可复现实验框架。项目统一管理数据集适配、论文方法、
模型调用、官方评分器、断点续跑和实验记录，同时将数据、方法源码与 runner 保持为
相互隔离的 Git 仓库。

## 功能

- 统一运行方法 × 数据集 × repeat 实验矩阵；
- 支持样本级断点续跑、并发调用、seed 和方法专属参数；
- 保存模型输出、评分结果、token usage、latency 和完整实验配置；
- 优先复用数据集官方 normalization、答案解析和等价检查；
- 使用固定数据文件和 Git object ID 标识实验数据；
- 提供确定性 smoke matrix，在调用正式 API 前验证完整执行链路。

## 支持范围

| 数据集 | 测试规模 | 评分方式 |
| --- | ---: | --- |
| GSM1K | 1,205 | 数值答案匹配 |
| MATH-Perturb | 230 | 官方 `answer_check` |
| HARP | 4,302 | 官方 `latex_answer_check` |
| HARP small | 1,434 | HARP test 的固定分层 1/3 子集 |
| U-MATH text-only | 720 | 固定配置的 LLM judge |
| MathConstruct | 439 | 官方 `parse_and_check` |

通用实验矩阵包含以下方法：

| 方法 | 类型 | 入口 |
| --- | --- | --- |
| `direct` | 单次直接回答基线 | `run_experiments.sh` |
| `zero_shot_cot` | 单次 zero-shot CoT 基线 | `run_experiments.sh` |
| `pal` | Program-Aided Language Models | `run_experiments.sh` |
| `self_refine` | Self-Refine | `run_experiments.sh` |
| `aflow` | validation workflow search + frozen test | `run_aflow.sh` |

## 工作区结构

数据和论文官方源码不复制进本仓库。默认工作区结构为：

```text
workspace/
├── benchmark-runner/
├── math-benchmark-data/
└── methods/
    ├── PAL/
    ├── Self-Refine/
    └── AFlow/
```

外部仓库来源和固定版本记录在 `manifests/repositories.toml`。可使用以下命令检查
工作区是否完整：

```bash
python3 scripts/verify_workspace.py
```

## 安装

需要 Python 3.11+：

```bash
python3 -m venv .venv-checkers
.venv-checkers/bin/pip install -r requirements/experiment-lock.txt
.venv-checkers/bin/pip install -e . --no-deps
cp .env.example .env
```

在 `.env` 中配置所用模型对应的 API key。程序只从环境变量读取凭据，`.env` 不会
进入 Git。

## 运行实验矩阵

```bash
./scripts/run_experiments.sh \
  --methods direct,pal,self_refine \
  --datasets gsm1k,math-perturb,harp-small,u-math-text-only \
  --repeats 3 \
  --batch-size all \
  --concurrency 4 \
  --temperature 0.7 \
  --seed 42 \
  --seed-mode increment
```

`--methods` 和 `--datasets` 可指定单项、逗号分隔列表或 `all`。`--batch-size N` 每次
顺序处理接下来的 N 条未完成样本，`--batch-size all` 处理全部剩余样本。
`--concurrency` 只改变并发调度，不改变实验身份。完整参数可直接查看：

```bash
./scripts/run_experiments.sh --help
```

各运行脚本的完整参数和记录说明见
[`docs/running_experiments.md`](docs/running_experiments.md)。

## 运行 AFlow

AFlow 搜索只使用对应数据集的固定 validation split，生成冻结 workflow 后再运行
test split。

```bash
# 搜索 workflow
./scripts/run_aflow.sh \
  --mode search \
  --dataset gsm1k \
  --model qwen35_flash \
  --optimizer-model qwen35_flash \
  --search-rounds 3 \
  --validation-size all

# 使用搜索产物测试
./scripts/run_aflow.sh \
  --mode test \
  --dataset gsm1k \
  --model qwen35_flash \
  --workflow <path-to-workflow.json> \
  --batch-size all \
  --repeats 3
```

## 评估 U-MATH judge

`run_mu_math.sh` 用官方 µ-MATH 数据比较候选 judge，不运行论文方法，也不会覆盖正式
U-MATH judge lock。

```bash
./scripts/run_mu_math.sh \
  --judge qwen37_flash \
  --batch-size all \
  --concurrency 4
```

内置候选包括 `qwen35_flash`、`qwen37_flash`、`gemini36_flash`、
`deepseek_v4_pro` 和 `deepseek_v4_flash`。

## 产物与续跑

新实验按模型和方法分层保存：

```text
results/experiments/<model>/<method>/
└── <dataset>__rNNN__<tag>__<UTC-time>__<fingerprint>/
    ├── experiment.json
    ├── records.jsonl
    ├── api_calls.jsonl
    ├── errors.jsonl
    └── summary.json
```

脚本通过 manifest 中的实验配置识别同一次实验，通过 `records.jsonl` 中的 problem ID
判断已完成样本。保持模型、方法、数据、repeat、seed、推理参数和 `run-tag` 不变即可
续跑；`batch-size` 和 `concurrency` 可以调整。结果目录不进入 Git。

## 测试

```bash
.venv-checkers/bin/pip check
.venv-checkers/bin/python -m unittest discover -v

MATH_CHECKER_PYTHON="$PWD/.venv-checkers/bin/python" \
  .venv-checkers/bin/python -m unittest tests.test_official_checkers -v

.venv-checkers/bin/python scripts/run_smoke_matrix.py
```

Smoke matrix 使用确定性 backend，不代表模型性能，但会实际执行方法适配、程序执行器
和官方 checker。

## 安全边界

PAL 和部分 Self-Refine 链路会执行模型生成的 Python。生成代码经过 AST 限制，并在
设置资源上限的独立子进程中运行；它不是面向任意不可信代码的完整安全沙箱。处理
不可信模型输出时，建议额外使用无网络容器或 microVM 隔离 worker。
