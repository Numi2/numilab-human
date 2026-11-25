#!/bin/bash
# Launch the integrated skin/fat release on its native variable-pleural-pressure owner.
set -euo pipefail
python="$HOME/numi-human-prep-venv-20261005/bin/python3"
owner="$(dirname "$0")/run_current_resting_contact.py"
integrated=(--dense --posterior-support conditioned-release)
case ${1:-} in
    "") exec "$python" "$owner" "${integrated[@]}" ;;
    --dense) shift; exec "$python" "$owner" --dense "$@" ;;
    --current|--regional) shift; exec "$python" "$owner" "$@" ;;
    --check-material)
        if (( $# != 2 )); then echo "Supply the material report output path" >&2; exit 2; fi
        exec "$python" "$owner" "${integrated[@]}" --material-report "$2"
        ;;
    --check|--steps|--seconds|-h|--help) exec "$python" "$owner" "${integrated[@]}" "$@" ;;
    16|60)
        if (( $# != 1 )); then echo "Expected one duration argument" >&2; exit 2; fi
        exec "$python" "$owner" "${integrated[@]}" --seconds "$1"
        ;;
    *) echo "Usage: $0 [--check|--steps N|16|60|--regional [OPTIONS]|--dense [OPTIONS]]" >&2; exit 2 ;;
esac
