#!/usr/bin/env bash
set -euo pipefail
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$app_dir/.venv/bin/python" ]]; then
  printf 'Missing local app environment. Follow the setup commands in README.md.\n' >&2
  exit 1
fi
exec "$app_dir/.venv/bin/python" -m streamlit run "$app_dir/app.py" "$@"
