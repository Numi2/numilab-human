#!/bin/bash
set -euo pipefail
case_dir=$(dirname "$(realpath "$0")")
parent=$(dirname "$case_dir")
if [[ "${1:-}" == "--help" ]]; then
    echo 'Run the 16-second supported native Human scene on Numi.'
    echo 'Usage: run-numi-human-rest [--forearm-contact | --material-contact] [new-output-directory]'
    echo 'Use --forearm-contact for the connected elbow skin and four loaded patches per forearm.'
    echo 'Use --material-contact to test dense-derived finite-strain bed contact. This scene exceeds its tissue strain envelope.'
    exit 0
fi
case "${1:-}" in
    --forearm-contact)
        source_dir="$case_dir/patch-source"
        owner_dir="$case_dir/patch-owner"
        ;;
    --material-contact)
        source_dir="$case_dir/finite-source"
        owner_dir="$case_dir/finite-owner"
        ;;
    *) exec "$parent/run-supported-human" "$@" ;;
esac
shift
if (( $# > 1 )); then exit 2; fi
if (( $# )); then
    out=$1
else
    out=$(mktemp -d "$case_dir/forearm-replay.XXXXXX")/run
fi
runner="$case_dir/run_bed_work.py"
python=/Users/numi/numi-human-prep-venv-20261005/bin/python3
export NUMI_RETAIN_SUPPORT_GEOMETRY=1 NUMI_CAPTURE_BED_COLUMNS=1
echo "Native evidence directory: $out"
exec "$python" "$runner" "$source_dir" "$owner_dir" "$out" 8000 --no-bed-work-audit --intermediate-steps 3999
