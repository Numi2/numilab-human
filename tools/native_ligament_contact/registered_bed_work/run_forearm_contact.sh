#!/bin/bash
set -euo pipefail
case_dir=$(dirname "$(realpath "$0")")
parent=$(dirname "$case_dir")
if [[ "${1:-}" == "--help" ]]; then
    echo 'Run the 16-second supported native Human scene on Numi.'
    echo 'Usage: run-numi-human-rest [--forearm-contact] [new-output-directory]'
    echo 'Use --forearm-contact for the connected elbow skin and four loaded patches per forearm.'
    exit 0
fi
if [[ "${1:-}" != "--forearm-contact" ]]; then
    exec "$parent/run-supported-human" "$@"
fi
shift
if (( $# > 1 )); then exit 2; fi
if (( $# )); then
    out=$1
else
    out=$(mktemp -d "$case_dir/forearm-replay.XXXXXX")/run
fi
runner="$parent/skin-publication/tools/native_ligament_contact/registered_bed_work/run_bed_work.py"
python=/Users/numi/numi-human-prep-venv-20261005/bin/python3
export NUMI_RETAIN_SUPPORT_GEOMETRY=1 NUMI_CAPTURE_BED_COLUMNS=1
echo "Native evidence directory: $out"
exec "$python" "$runner" "$case_dir/patch-source" "$case_dir/patch-owner" "$out" 8000 --no-bed-work-audit --intermediate-steps 3999
