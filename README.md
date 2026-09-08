# Reproducible Evaluation of Mathematical Reasoning Methods

This repository is the orchestration layer for a controlled comparison of
inference-time reasoning methods on mathematical benchmarks. It keeps the
evaluation code, benchmark data, and upstream method implementations in
separate Git repositories, while recording enough provenance to trace every
result back to a model configuration, dataset snapshot, scorer, and source
revision.

The project is designed for method comparison rather than prompt optimization.
Each method retains its own reasoning procedure. The runner adds only the
answer format required by a benchmark, then evaluates the response with that
benchmark's scorer or, for U-MATH, a separately fixed judge.

## What is implemented

- method × dataset × repeat experiment matrices;
- sample-level resume with durable per-sample writes;
- configurable concurrency, sampling parameters, seeds, and method settings;
- per-call latency, token usage, retry, and request metadata;
- dataset-specific response formatting paired with dataset-specific scoring;
- pinned data files and upstream method revisions;
- a deterministic smoke matrix that exercises adapters, code execution, and
  official checkers without calling a model API;
- a separate AFlow search-and-test pipeline that keeps validation search apart
  from final test evaluation.

## Evaluation scope

### Benchmarks

| Dataset | Test examples | Evaluation |
| --- | ---: | --- |
| GSM1K | 1,205 | normalized numeric answer |
| MATH-Perturb | 230 | upstream `answer_check` |
| HARP | 4,302 | upstream `latex_answer_check` |
| HARP small | 1,434 | fixed stratified one-third subset of the HARP test split |
| U-MATH text-only | 720 | locked LLM judge configuration |
| MathConstruct | 439 | upstream `parse_and_check` |

The data repository also contains fixed validation splits for methods that
require model or workflow selection. Validation and test membership is frozen
before an experiment starts; the runner does not resample either split at run
time.

### Methods

| Method | Role | Entry point |
| --- | --- | --- |
| `direct` | single-pass direct answer baseline | `run_experiments.sh` |
| `zero_shot_cot` | single-pass zero-shot chain-of-thought baseline | `run_experiments.sh` |
| `pal` | Program-Aided Language Models | `run_experiments.sh` |
| `self_refine` | iterative feedback and revision | `run_experiments.sh` |
| `aflow` | validation-time workflow search followed by frozen testing | `run_aflow.sh` |

PAL, Self-Refine, and AFlow are integrated from pinned upstream snapshots. The
source repositories, papers, licenses, and exact revisions are listed in
[`manifests/repositories.toml`](manifests/repositories.toml).

## Repository layout

The default workspace contains four independent Git boundaries:

```text
workspace/
├── benchmark-runner/          # this repository
├── math-benchmark-data/       # prepared data, splits, and official checkers
└── methods/
    ├── PAL/
    ├── Self-Refine/
    └── AFlow/
```

This separation prevents local changes to a paper implementation or dataset
snapshot from being hidden inside a runner commit. Check the complete workspace
against the pinned manifest with:

```bash
python3 scripts/verify_workspace.py
```

## Installation

Python 3.11 or later is required. The dedicated environment includes the
runtime dependencies used by the official checkers.

```bash
python3 -m venv .venv-checkers
.venv-checkers/bin/pip install -r requirements/experiment-lock.txt
.venv-checkers/bin/pip install -e . --no-deps
cp .env.example .env
```

Add only the API keys needed by the selected model profile to `.env`. Credentials
are read from environment variables; `.env` and generated results are excluded
from Git.

## Running an experiment

The following command evaluates three methods on four test sets, with three
repeats per cell:

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

Both `--methods` and `--datasets` accept one name, a comma-separated list, or
`all`. Use `--batch-size N` to process the next `N` unfinished examples in each
cell, or `--batch-size all` to finish every remaining example. Concurrency and
batch size may be changed between resumed runs without changing experiment
identity.

Run a dry check before a long experiment:

```bash
./scripts/run_experiments.sh \
  --methods pal \
  --datasets gsm1k,math-perturb,harp-small,u-math-text-only \
  --batch-size 20 \
  --dry-run
```

The complete command reference, including method parameters, resume semantics,
failure handling, AFlow, and µ-MATH judge evaluation, is in
[`docs/running_experiments.md`](docs/running_experiments.md).

## AFlow workflow search

AFlow is intentionally separate from the main method matrix. Search mode reads
only the fixed validation split and writes a frozen workflow. Test mode requires
that workflow as an explicit input.

```bash
# Search on validation data.
./scripts/run_aflow.sh \
  --mode search \
  --dataset gsm1k \
  --model qwen35_flash \
  --optimizer-model qwen35_flash \
  --search-rounds 3 \
  --validation-size all

# Evaluate the selected workflow on the test split.
./scripts/run_aflow.sh \
  --mode test \
  --dataset gsm1k \
  --model qwen35_flash \
  --workflow <path-to-workflow.json> \
  --batch-size all \
  --repeats 3
```

The workflow records its validation dataset and search configuration. Test mode
rejects a workflow associated with a different benchmark.

## U-MATH judge selection

Free-form U-MATH answers are evaluated by a judge whose profile is locked
independently of the model under test. Candidate judges can be measured on the
official µ-MATH meta-evaluation set without changing that lock:

```bash
./scripts/run_mu_math.sh \
  --judge qwen37_flash \
  --batch-size all \
  --concurrency 4
```

The script reports classification metrics overall and by answer-source model.
Judge profile settings are fixed in TOML rather than inherited from the tested
method.

## Artifacts and provenance

Experiment outputs are grouped by model and method:

```text
results/experiments/<model>/<method>/
└── <dataset>__rNNN__<tag>__<UTC-time>__<fingerprint>/
    ├── experiment.json
    ├── records.jsonl
    ├── api_calls.jsonl
    ├── errors.jsonl          # present only when an error was recorded
    └── summary.json
```

`experiment.json` stores the effective configuration and provenance.
`records.jsonl` is the durable sample ledger used for resume. A repeated command
finds the matching experiment from its stable identity and skips problem IDs
already present in that ledger. Failed API, environment, or scoring operations
do not create a completed sample record, so the same command can continue after
the cause is fixed.

Generated outputs are deliberately not committed: they may be large and can
contain full model responses. The manifest and JSONL files are retained locally
as the audit trail for analysis.

## Verification

```bash
.venv-checkers/bin/pip check
.venv-checkers/bin/python -m unittest discover -v

MATH_CHECKER_PYTHON="$PWD/.venv-checkers/bin/python" \
  .venv-checkers/bin/python -m unittest tests.test_official_checkers -v

.venv-checkers/bin/python scripts/run_smoke_matrix.py
```

The smoke matrix uses deterministic local responses. It validates the execution
path, not model quality.

## Limitations and safety

This repository evaluates inference-time behavior; it is not a training
framework. Results remain sensitive to model-provider changes even when the
request configuration is fixed, so model IDs, request metadata, and timestamps
are preserved with each run.

PAL and part of the Self-Refine pipeline execute model-generated Python. The
executor applies AST restrictions and runs code in a resource-limited child
process, but it is not a complete sandbox for arbitrary untrusted programs. For
untrusted endpoints, run the worker in an additional network-isolated container
or microVM.
