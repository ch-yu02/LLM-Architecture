#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runner_root="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${runner_root}/.venv-checkers/bin/python"

if [[ ! -x "${python_bin}" ]]; then
  echo "Missing benchmark environment: ${python_bin}" >&2
  exit 2
fi

exec "${python_bin}" "${script_dir}/run_aflow.py" "$@"
