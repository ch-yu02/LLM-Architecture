# Running experiments

The repository exposes three commands:

| Command | Purpose |
| --- | --- |
| `scripts/run_experiments.sh` | evaluate the standard method × dataset × repeat matrix |
| `scripts/run_aflow.sh` | search for an AFlow workflow on validation data, or test a frozen workflow |
| `scripts/run_mu_math.sh` | compare candidate U-MATH judges on µ-MATH |

Run all commands from the repository root. In multiline shell examples, each
backslash must be the final character on its line.

## Environment

Create the dedicated Python 3.11+ environment and install the pinned experiment
dependencies:

```bash
python3 -m venv .venv-checkers
.venv-checkers/bin/pip install -r requirements/experiment-lock.txt
.venv-checkers/bin/pip install -e . --no-deps
cp .env.example .env
```

Set the keys needed by the profiles you plan to use:

```dotenv
DASHSCOPE_API_KEY_BEIJING=
DEEPSEEK_API_KEY=
GEMINI_API_KEY=
```

Do not commit `.env`. The runner reads credentials from environment variables
and does not copy them into public experiment manifests.

Before a long run, verify the Python environment, external repositories, data,
and checker integration:

```bash
.venv-checkers/bin/pip check
.venv-checkers/bin/python scripts/verify_workspace.py
.venv-checkers/bin/python ../math-benchmark-data/scripts/validate_datasets.py
MATH_CHECKER_PYTHON="$PWD/.venv-checkers/bin/python" \
  .venv-checkers/bin/python -m unittest discover -v
```

## Batches, resume, and concurrency

`--batch-size N` processes the next `N` unfinished examples in each experiment
cell. `--batch-size all` processes all remaining examples. A cell is one method,
dataset, and repeat combination.

```bash
--batch-size 100
--batch-size all
```

On a resumed run, the script locates an experiment with the same stable identity
and reads its `records.jsonl`. Problem IDs already present are skipped. The
confirmation screen reports only the maximum number of new samples.

The identity includes the model and inference settings, method configuration,
dataset and split, repeat, seed, answer protocol, relevant dataset and checker
files, judge configuration where applicable, run tag, and the relevant Python
environment. Runner revisions remain in the manifest for audit, but an unrelated
documentation commit does not invalidate existing progress.

The following options control execution but do not change experiment identity:

- `--batch-size`
- `--concurrency`
- `--yes`
- `--debug`
- `--skip-preflight`

`--concurrency N` allows at most `N` sample requests to be in flight within the
current cell; the default is `1`. Cells themselves are evaluated sequentially.
Changing concurrency between runs does not split the experiment.

Each model profile may also define `min_request_interval_seconds`. A value of
`1.0` spaces request starts by at least one second even when concurrency is
higher. Reduce it only when the provider's rate limit permits, for example:

```toml
min_request_interval_seconds = 0.2
```

## Standard method matrix

### Basic commands

With no arguments, the script runs the next unfinished `direct × gsm1k × repeat
1` sample:

```bash
./scripts/run_experiments.sh
```

Inspect a matrix without making API calls or creating experiment artifacts:

```bash
./scripts/run_experiments.sh \
  --methods direct,pal,self_refine \
  --datasets gsm1k,math-perturb,harp-small,u-math-text-only \
  --repeats 3 \
  --batch-size 20 \
  --concurrency 4 \
  --dry-run
```

Run the first 100 unfinished examples in each cell:

```bash
./scripts/run_experiments.sh \
  --methods pal,self_refine \
  --datasets gsm1k,math-perturb,harp-small,u-math-text-only \
  --repeats 3 \
  --batch-size 100 \
  --concurrency 8 \
  --temperature 0.7 \
  --seed 42 \
  --seed-mode increment
```

Rerun the same command with `--batch-size all` to finish the remaining examples.

### Methods and datasets

`--methods` accepts:

```text
direct, zero_shot_cot, pal, self_refine, all
```

AFlow is not part of this matrix; use `run_aflow.sh` instead.

`--datasets` accepts:

| Name | Test examples | Evaluator |
| --- | ---: | --- |
| `gsm1k` | 1,205 | normalized numeric answer |
| `math-perturb` | 230 | upstream `answer_check` |
| `harp` | 4,302 | upstream `latex_answer_check` |
| `harp-small` | 1,434 | fixed HARP test subset; upstream `latex_answer_check` |
| `u-math-text-only` | 720 | locked LLM judge |
| `mathconstruct` | 439 | upstream `parse_and_check` |

`--datasets all` selects the five full benchmarks and excludes `harp-small`,
which is a subset of HARP. Select `harp-small` explicitly when using the reduced
test set. Validation sets cannot be selected through this command.

The U-MATH judge is locked to `qwen3.7-flash-2026-07-15` through
`configs/judges/u_math.lock.toml`. Its temperature, seed, and other inference
settings do not inherit values from the model being evaluated.

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--model MODEL` | `qwen35_flash` | profile name, configured model ID, or TOML path |
| `--model-id ID` | profile value | override the model ID while retaining the profile |
| `--methods LIST` | `direct` | one method, a comma-separated list, or `all` |
| `--datasets LIST` | `gsm1k` | one dataset, a comma-separated list, or `all` |
| `--repeats N` | `1` | number of repeats for every method × dataset cell |
| `--batch-size N\|all` | `1` | new examples to process per cell in this invocation |
| `--concurrency N` | `1` | concurrent sample requests within a cell |
| `--temperature X` | profile value | sampling temperature, from `0` to `2` |
| `--top-p X` | profile value | nucleus sampling threshold, from `0` to `1` |
| `--max-output-tokens N` | profile value | maximum output tokens per model call |
| `--seed N` | profile value or omitted | base seed, from `0` to `2147483647` |
| `--seed-mode MODE` | `fixed` | `fixed` or `increment` |
| `--method-param SPEC` | none | override one method setting as `METHOD.KEY=JSON_VALUE` |
| `--method-config SPEC` | none | load a method config as `METHOD=/path/config.toml` or JSON |
| `--run-tag TAG` | empty | human-readable label included in experiment identity |
| `--data-root PATH` | `../math-benchmark-data` | prepared data repository |
| `--output-root PATH` | `results/experiments` | experiment artifact root |
| `--checker-python PATH` | `.venv-checkers/bin/python` | interpreter used by official checker bridges |
| `--env-file PATH` | `.env` | environment-variable file |
| `--dry-run` | off | validate and display the plan without API calls |
| `--yes` | off | skip the interactive confirmation |
| `--skip-preflight` | off | bypass the initial API connectivity check |
| `--debug` | off | print full tracebacks as well as recording diagnostics |

Use `--skip-preflight` only when a separate connectivity check has already been
performed. It does not disable error handling during the experiment.

### Seeds and repeats

```bash
./scripts/run_experiments.sh \
  --methods direct \
  --datasets gsm1k \
  --repeats 3 \
  --seed 1234 \
  --seed-mode increment \
  --batch-size all
```

With `fixed`, every repeat uses the same seed. With `increment`, repeat `n` uses
`base_seed + n - 1`; a base seed is required. A provider may still introduce
nondeterminism, so a fixed seed is not a guarantee of identical responses. If
temperature is zero and the provider is deterministic, repeated runs with the
same seed may offer little additional information.

### Method-specific settings

`--method-param` can be supplied more than once. Values are parsed as JSON and
applied after any file supplied through `--method-config`.

```bash
./scripts/run_experiments.sh \
  --methods self_refine \
  --datasets gsm1k \
  --method-param self_refine.max_refinements=2 \
  --batch-size all
```

Method settings are part of experiment identity. Changing one starts a distinct
experiment rather than resuming an incompatible run.

### Output files

```text
results/experiments/
└── <model>/
    └── <method>/
        ├── <dataset-a>__r001__<tag>__<UTC-time>__<fingerprint>/
        ├── <dataset-a>__r002__<tag>__<UTC-time>__<fingerprint>/
        └── <dataset-b>__r001__<tag>__<UTC-time>__<fingerprint>/
```

Each cell contains:

| File | Contents |
| --- | --- |
| `experiment.json` | effective configuration, environment, revisions, and fingerprint |
| `records.jsonl` | problem, reference answer, metadata, model output, score, usage, and timing |
| `api_calls.jsonl` | request latency, token counts, retries, seed, and provider request ID |
| `errors.jsonl` | durable diagnostics, created only after an error |
| `summary.json` | aggregate accuracy, counts, usage, and timing |

A `method_failed` record means the method completed its model interaction but
did not produce a scoreable answer. It counts as incorrect and evaluation
continues. API failures, unavailable execution dependencies, and scorer failures
stop the run without marking that sample complete. After correcting the cause,
repeat the same command to continue from the last completed problem.

## AFlow search and test

`run_aflow.sh` handles one dataset at a time. It supports `gsm1k`,
`math-perturb`, `harp`, and `u-math-text-only`; it does not support HARP small or
MathConstruct.

### Search mode

Search mode evaluates a declarative workflow on the dataset's fixed validation
split. Each round proposes one candidate. The selected workflow maximizes
validation accuracy, with fewer model calls and fewer nodes used as tie-breakers.

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

Use `--validation-size 10` for a pipeline check. The optimizer defaults to the
execution model. Its generation settings are controlled separately by
`--optimizer-temperature`, `--optimizer-max-output-tokens`, and
`--optimizer-seed`.

Search artifacts are stored under:

```text
results/aflow/<execution-model>/aflow/
└── search__<dataset>__<UTC-time>__<fingerprint>/
    ├── search.json
    ├── search_state.json
    ├── optimizer_api_calls.jsonl
    ├── candidates/round-*/
    └── workflow.json
```

### Test mode

`--workflow` must point to a generated `workflow.json` or its containing search
directory. The runner verifies that the workflow is frozen, was not optimized on
test data, and belongs to the selected dataset.

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

Test runs resume at sample level. Their directories sit beside the corresponding
search directories under
`results/aflow/<execution-model>/aflow/<dataset>__rNNN__.../`.
`--batch-size`, `--repeats`, and `--seed-mode` apply only in test mode;
`--search-rounds` and `--validation-size` apply only in search mode.

Run `./scripts/run_aflow.sh --help` for the full option list.

## Evaluating U-MATH judges with µ-MATH

The official µ-MATH test data contains 271 problems and one candidate answer
from each of four source models, for 1,084 labeled rows. This command evaluates
the judge only; it does not run a reasoning method or modify the locked U-MATH
judge.

Check the configuration without making API calls:

```bash
./scripts/run_mu_math.sh \
  --judge qwen37_flash \
  --batch-size 10 \
  --concurrency 4 \
  --dry-run
```

Complete the evaluation:

```bash
./scripts/run_mu_math.sh \
  --judge qwen37_flash \
  --batch-size all \
  --concurrency 8
```

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--judge JUDGE` | `qwen35_flash` | candidate profile name, model ID, or TOML path |
| `--batch-size N\|all` | `1` | new labeled rows to process |
| `--concurrency N` | `1` | concurrent judge requests |
| `--data-root PATH` | `../math-benchmark-data` | prepared data repository |
| `--output-root PATH` | `results/mu_math` | result root |
| `--env-file PATH` | `.env` | environment-variable file |
| `--dry-run` | off | validate and display the plan without API calls |
| `--yes` | off | skip the interactive confirmation |
| `--skip-preflight` | off | bypass the initial API connectivity check |
| `--debug` | off | print full tracebacks in addition to recording diagnostics |

Model ID, temperature, seed, and output length cannot be overridden on this
command line. They belong to the candidate profile, which keeps judge
comparisons reproducible.

### Candidate profiles

| Profile | Model | Credential |
| --- | --- | --- |
| `qwen35_flash` | `qwen3.5-flash-2026-02-23` | `DASHSCOPE_API_KEY_BEIJING` |
| `qwen37_flash` | `qwen3.7-flash-2026-07-15` | `DASHSCOPE_API_KEY_BEIJING` |
| `gemini36_flash` | `gemini-3.6-flash`, reasoning effort `medium` | `GEMINI_API_KEY` |
| `deepseek_v4_pro` | `deepseek-v4-pro`, thinking enabled, reasoning effort `high` | `DEEPSEEK_API_KEY` |
| `deepseek_v4_flash` | `deepseek-v4-flash`, thinking enabled, reasoning effort `high` | `DEEPSEEK_API_KEY` |

The Qwen profiles fix `temperature=0`, `top_p=1`, and `seed=20260729`. Gemini
and DeepSeek thinking profiles omit temperature, top-p, and seed. Every bundled
candidate uses `max_output_tokens=4096`.

To test another judge without changing an existing profile, copy one to a new
file and select that path:

```bash
cp configs/judges/candidates/qwen35_flash.toml \
  configs/judges/candidates/my_candidate.toml

./scripts/run_mu_math.sh \
  --judge configs/judges/candidates/my_candidate.toml \
  --batch-size all
```

The judge returns `Yes`, `No`, or `Inconclusive`. Invalid output is treated as
`Inconclusive`, and `Inconclusive` counts as incorrect. `summary.json` reports
macro-F1, positive and negative F1, accuracy, TPR, TNR, PPV, NPV, the confusion
matrix, inconclusive count, token use, latency, retries, and errors. Metrics are
reported overall and by answer-source model.

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

## Failures and exit codes

Sample diagnostics are appended to `errors.jsonl`. With `--debug`, the same
traceback is also printed in the terminal. An uncaught top-level exception is
written to `_fatal_errors.jsonl` under the selected output root.

| Exit code | Meaning |
| ---: | --- |
| `0` | completed, already complete, or dry-run finished |
| `1` | a sample-level error stopped the run |
| `2` | invalid arguments, configuration, repository, data, environment, or preflight |
| `130` | interrupted by the user; completed sample records remain resumable |

For the authoritative option list installed in the current checkout, use:

```bash
./scripts/run_experiments.sh --help
./scripts/run_aflow.sh --help
./scripts/run_mu_math.sh --help
```
