#!/bin/bash
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=$(mktemp -d)
work="$HOME/Work/human-integrated-gaps-20261104"
src="$work/resting-surface-clearance-001/src"
root="$HOME/numi-human-retained-delivery-20261009/anatomy-completion-1276"
atlas="$root/vein-owner-blend-source-append-dryrun-012-inputs/Sources/isa_BP3D_4.0_obj_99.zip"
author="$root/full-anatomy-delivery-audit-1378/prepare_aortic7_parent7_rest_registration_reemission_001.py"
export PATH="/opt/homebrew/bin:$PATH"
export PYTHONPATH="$here:$HOME/Work/skin-fat-native-integration-209/tools:$work/resting-clearance-and-atrium-005:$src"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
zstd -dc "$here/accepted-body-poses.csv.zst" > "$out/poses.csv"
args=("$here/supported-run.json" "$here/reference-anchors.json" "$out/poses.csv")
"$py" "$here/compare_atlas.py" "${args[@]}" "$atlas" "$out/source.json"
"$py" "$here/compare_registration.py" "${args[@]}" "$atlas" "$out/default.json"
cmp "$out/default.json" "$here/default-registration.json"
"$py" "$here/fit_articular_frame.py" "${args[@]}" "$atlas" "$out/frames.json"
cmp "$out/frames.json" "$here/articular-frame.json"
"$py" "$here/audit_joint_neighbors.py" "${args[@]}" "$out/frames.json" "$out/neighbors.json"
cmp "$out/neighbors.json" "$here/joint-neighbors.json"
"$py" "$here/check_vessel_transfer.py" "$here/supported-run.json" "$here/joint-source-bound.nhrigid" "$author" "$out/vessels.json"
cmp "$out/vessels.json" "$here/vessel-transfer.json"
echo "Verified retained shoulder geometry and vessel transfer at $out"
