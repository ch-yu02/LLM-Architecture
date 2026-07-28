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
| PAL | 固定源码快照，tree 已校验 | `scripts/gsm_eval.py` | 未安装 |
| Self-Refine | 官方 Git clone | `src/gsm/run.py` | 未安装；依赖 `prompt-lib` |
| AFlow | 官方 Git clone | `run.py --dataset MATH` | 未安装；要求 Python 3.9 |

`source_ready` 仅表示官方源码、版本和入口已经准备好。三种方法尚未统一接入
GSM1K、MATH-Perturb、HARP、U-MATH text-only 和 MathConstruct，也尚未配置具体
模型 API。这样可以避免把“源码已下载”误认为“实验可直接复现”。

## 验证源码版本

脚本只依赖 Python 3.11+ 标准库：

```bash
python3 scripts/verify_workspace.py
```

版本、官方地址、许可、入口与额外依赖记录在
`manifests/repositories.toml`。每种方法的集成计划位于
`configs/methods/`。

## 仓库约定

- 对官方方法的修改放在各方法仓库的 `benchmark-integration` 分支。
- 官方 remote 使用 `upstream`；以后将个人 fork 添加为 `origin`。
- 本仓库只跟踪 adapter、实验配置和小型汇总结果。
- 完整模型输出放入 `results/raw/`，该目录默认不进入 Git。
- 所有实验必须同时记录 data、runner、method 和 model revision。

## 安全提示

PAL 会执行模型生成的 Python 程序。正式实验前必须将执行后端放进受限容器或沙箱，
不要直接在包含密钥和用户文件的主进程中执行模型输出。

