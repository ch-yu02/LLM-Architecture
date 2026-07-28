# Math benchmark runner

本仓库统一管理数学评测数据版本、论文方法源码版本和后续实验配置。论文官方源码
保持在相邻的独立仓库中，不复制到本仓库。

```text
LLM_architecture/
├── math-benchmark-data/
├── benchmark-runner/
└── methods/
    ├── PAL/
    ├── Self-Refine/
    └── AFlow/
```

## 当前状态

| 方法 | 源码状态 | 数学入口 | 环境状态 |
| --- | --- | --- | --- |
| PAL | 固定源码快照，tree 已校验 | `PALAdapter` | runtime ready |
| Self-Refine | 官方 Git clone | `SelfRefineAdapter` | runtime ready |
| AFlow | 官方 Git clone | `AFlowAdapter` | frozen workflow runtime ready |

三种方法均已接入共享 `ModelBackend`，无需加载论文仓库中的旧模型 client。
正式实验默认使用 Qwen3.5 Flash profile，也可从命令行切换兼容模型。接入语义、
预算和安全边界见
[`docs/method_adapters.md`](docs/method_adapters.md)。

统一评测核心和五个数据集插件已经建立：

| 数据集 | loader | scorer |
| --- | --- | --- |
| GSM1K | unified JSONL | 数值答案 |
| MATH-Perturb | 固化 test JSONL（230 题） | 官方 `answer_check` |
| HARP | 固化 test JSONL（4,302 题） | 官方 `latex_answer_check` |
| U-MATH text-only | 固化 test JSONL（720 题） | 显式注入的 LLM judge |
| MathConstruct | 固化 JSONL（97 families / 439 题） | 官方 `parse_and_check` |

runner 只依赖 `DatasetPlugin` 协议，不包含数据集名称分支。新增数据集方法见
[`docs/adding_dataset.md`](docs/adding_dataset.md)。

## 公平基础方法

当前包含两个公平基础方法：

| 方法 | 模型调用/题 | few-shot | 工具 |
| --- | ---: | --- | --- |
| `direct` | 1 | 否 | 否 |
| `zero_shot_cot` | 1 | 否 | 否 |

二者共享模型与推理参数，只改变求解策略指令。runner 强制调用预算，记录实际调用
配置、latency 和 usage，并阻止同一 experiment ID 混入不同配置。公共配置位于
`configs/evaluation/fair_baselines.toml`，详细约束见
[`docs/fair_baselines.md`](docs/fair_baselines.md)。

## Smoke matrix

仓库提供固定的 5 数据集 × 5 方法 plumbing matrix：

```bash
.venv-checkers/bin/python scripts/run_smoke_matrix.py
```

矩阵使用确定性 canned backend，不代表模型性能；三套官方 checker 和方法执行链仍
会真实运行。定义、范围与报告位置见
[`docs/smoke_matrix.md`](docs/smoke_matrix.md)。

## 运行正式实验

默认模型配置为 `qwen3.5-flash-2026-02-23`，API key 仅从环境变量读取。安装实验
依赖、配置 key 后可直接运行：

```bash
.venv-checkers/bin/pip install -r requirements/experiment-lock.txt
.venv-checkers/bin/pip install -e . --no-deps
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY_BEIJING
./scripts/run_experiments.sh \
  --methods direct \
  --datasets gsm1k \
  --repeats 3 \
  --batch-size 1 \
  --seed 1234 \
  --seed-mode increment
```

脚本支持方法 × 数据集 × 重复轮次矩阵、样本级断点续跑、方法专属参数以及实时
latency/token/进度显示。`--seed` 会传给所有模型调用并纳入实验指纹；
`--seed-mode increment` 使各 repeat 依次使用 `seed + repeat_index - 1`，默认
`fixed` 则使用相同 seed。不指定 seed 时不发送该参数。实验记录使用单层目录：

```text
results/experiments/
└── qwen35-flash-bj__direct__gsm1k__r001__<fingerprint>/
    ├── experiment.json
    ├── records.jsonl
    ├── api_calls.jsonl
    ├── errors.jsonl        # 仅发生样本错误时生成
    └── summary.json
```

`records.jsonl` 使用紧凑格式：首行只记录一次 experiment identity；样本行保留
题目正文、参考答案、dataset metadata、模型输出、评分、token、耗时、模型调用数
和必要的方法产物。完整实验配置与 revision 统一保存在 `experiment.json`，不会在
每条样本中重复。

完整命令与记录协议见
[`docs/running_experiments.md`](docs/running_experiments.md)。

U-MATH 的正式 judge 将由 `configs/judges/u_math.lock.toml` 唯一锁定，
不提供命令行覆盖能力。当前 lock 状态为 `pending`：在 µ-MATH 候选测试完成并
选定 judge 前，正式 U-MATH 实验会拒绝启动，避免提前产生评分协议不一致的结果。
候选 profile 放在 `configs/judges/candidates/`，其中 Qwen3.5-Flash 目前只是一项
候选，并不是已经选定的正式 judge。锁定后，judge 仍不会继承被测模型、
`--model-id`、实验 temperature 或 repeat seed。

项目采用 U-MATH 官方的 manual CoT 判断流程：prompt 要求 judge 依次抽取候选
答案、完成必要变换、比较参考答案并给出最终 verdict。按本项目约定，官方独立
`Qwen2.5-72B` extractor 被移除，judge 自己必须在最后一行输出
`Yes`、`No` 或 `Inconclusive`。本地确定性解析器只读取最后一个独立 verdict；
如果输出不合规，则回退为 `Inconclusive`。只有 `Yes` 计为正确，`No` 和
`Inconclusive` 都计为错误。

该协议记为 `u-math-manual-cot-self-verdict`，属于“官方 manual CoT + 自判三态”
变体，不能声称逐项复刻带 Qwen2.5-72B extractor 的原始官方流水线。正式实验会在
manifest 和样本记录中保存协议版本、judge profile、三态 verdict、解析状态、
latency 和 token usage。官方 µ-MATH 的比较仍以 macro-F1 为主，并应同时报告
TPR、TNR、PPV、NPV 及候选答案来源模型切片；普通 U-MATH 实验汇总 accuracy。

用于比较 judge 的官方 µ-MATH 数据位于数据仓库
`data/processed/mu_math.jsonl`，论文 Table 5 的结构化结果位于
`data/mu_math/official_results.json`。

## 环境与测试

runner、API client 和官方 checker 统一安装在 runner 自己的
`.venv-checkers`；三个论文源码仓库不安装到该环境，也不会彼此导入。官方方法只通过
本仓库 adapter 读取固定 prompt/workflow，生成程序则进入受限子进程。推荐使用已经
验证的依赖锁：

```bash
python3 -m venv .venv-checkers
.venv-checkers/bin/pip install -r requirements/experiment-lock.txt
.venv-checkers/bin/pip install -e . --no-deps
.venv-checkers/bin/pip check
.venv-checkers/bin/python -m unittest discover -v
MATH_CHECKER_PYTHON="$PWD/.venv-checkers/bin/python" \
  .venv-checkers/bin/python -m unittest tests.test_official_checkers -v
```

默认测试覆盖统一 schema、紧凑 JSONL 结果落盘、顺序分批续跑、experiment 冲突
检测、正式 CLI 的本地 OpenAI-compatible 5×5 执行链、基础方法公平预算、插件
扩展契约、四个静态数据集的实际样本数，以及 GSM1K/U-MATH 的评分行为。设置
`MATH_CHECKER_PYTHON` 后额外运行三个原生 checker 的集成测试。

## 验证源码版本

脚本只依赖 Python 3.11+ 标准库：

```bash
python3 scripts/verify_workspace.py
```

数据仓库的路径与分支、论文方法的固定版本、官方地址、许可、入口和额外依赖记录在
`manifests/repositories.toml`。数据仓库允许独立演进；每次实验会把其实际 commit
写入 `experiment.json`，而不是在 runner 中硬编码数据 HEAD。每种方法的运行配置
位于 `configs/methods/`。

## 仓库约定

- 对官方方法的修改放在各方法仓库的 `benchmark-integration` 分支。
- 官方 remote 使用 `upstream`；以后将个人 fork 添加为 `origin`。
- 本仓库只跟踪 adapter、实验配置和测试；正式实验产物不进入 Git。
- 完整实验产物放入 `results/experiments/`，该目录默认不进入 Git。
- 所有实验必须同时记录 data、runner、method 和 model revision。

## 安全提示

PAL 和 Self-Refine 会执行模型生成的 Python 程序。当前 adapter 使用受限 AST、
独立子进程和资源上限，不在 runner 主进程执行。面对不受信任模型的正式实验，仍
建议将 worker 再放入无网络容器或 microVM。
