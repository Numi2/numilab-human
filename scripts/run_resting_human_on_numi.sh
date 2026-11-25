#!/bin/bash
# Launch the supported, variable-pleural-pressure native scene on SSH Numi.
set -euo pipefail
python="$HOME/numi-human-prep-venv-20261005/bin/python3"
owner="$(dirname "$0")/run_current_resting_contact.py"
case ${1:-16} in
    --dense) shift; exec "$python" "$owner" --dense "$@" ;;
    --current) shift; exec "$python" "$owner" "$@" ;;
    --check-material)
        if (( $# != 2 )); then echo "Supply the material report output path" >&2; exit 2; fi
        exec "$python" "$owner" --material-report "$2"
        ;;
    --check) shift; exec "$python" "$owner" --check "$@" ;;
    16|60)
        if (( $# > 1 )); then echo "Expected one duration argument" >&2; exit 2; fi
        exec "$python" "$owner" --seconds "${1:-16}"
        ;;
    *) echo "Usage: $0 [16|60|--check|--check-material REPORT|--current [--check|--steps N]|--dense [--check|--steps N]]" >&2; exit 2 ;;
esac
