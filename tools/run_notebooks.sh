#!/usr/bin/env bash
# Execute every library notebook (skips run_*.ipynb, which need hardware/GPU/data).
# usage: tools/run_notebooks.sh [path ...]     (default: the whole repo)
# needs: pip install nbclient ipykernel
set -u
cd "$(dirname "$0")/.."
fail=0
mapfile -t nbs < <(find "${@:-.}" -name '*.ipynb' -not -name 'run_*' \
                    -not -path '*/.ipynb_checkpoints/*' -not -path './.git/*' | sort)
for nb in "${nbs[@]}"; do
  if (cd "$(dirname "$nb")" && jupyter-execute --timeout=300 "$(basename "$nb")" >/dev/null 2>/tmp/nb_err.txt); then
    echo "OK    $nb"
  else
    echo "FAIL  $nb"; tail -n 5 /tmp/nb_err.txt | sed 's/^/      /'; fail=1
  fi
done
exit $fail
