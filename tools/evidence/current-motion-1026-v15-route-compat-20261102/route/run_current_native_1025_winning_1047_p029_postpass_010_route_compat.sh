#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 RUN_MANIFEST SOURCE_CLOSURE BUILD_MANIFEST CHECKPOINT [JOB_DIR]" >&2
  exit 64
fi
RUN=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
CLOSURE=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
BUILD=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
CHECKPOINT=$(cd "$(dirname "$4")" && pwd)/$(basename "$4")
RUN_DIR=$(dirname "$RUN")
JOB=${5:-$RUN_DIR/native-009-integration}
PY=/Users/numi/numi-human-prep-venv-20261005/bin/python3
P=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-native-1025-launch/current-native-1025-direct-o-007-recorded-flow-scope
CONTRACT=$P/recorded_flow_contract_009_postpass.py
IDENTITY=$P/materialize_current_fractional_013_identity_009.py
MASS_BUILDER=$P/build_current_fractional_mass_receipt_013_nested_controls_compat_009.py
MATERIALIZE_RECORDED=$P/materialize_current_native_1025_direct_o_009_recorded_flow_dir.py
PREPARE=$P/prepare_current_native_1025_direct_o_009.py
CONVERT=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-fractional-recorded-flow-1050/solver-free-converter-001/convert_current_fractional_combined_flow.py
CONVERTER=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-fractional-recorded-flow-1050/solver-free-converter-001/convert-current-fractional-combined-flow
MOTION_VERIFY=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-motion-1026-verifier/verify_current_motion_geometry_1026_v15.py
EXPORTER=/Users/numi/Work/skin-fat-native-integration-209/build-native/numi-human-accepted-skin-export
for path in "$RUN" "$CLOSURE" "$BUILD" "$CHECKPOINT" "$PY" "$CONTRACT" "$IDENTITY" "$MASS_BUILDER" "$MATERIALIZE_RECORDED" "$PREPARE" "$CONVERT" "$CONVERTER" "$MOTION_VERIFY" "$EXPORTER"; do
  [[ -f "$path" ]] || { echo "required input is missing: $path" >&2; exit 2; }
done
[[ ! -e "$JOB" ]] || { echo "refusing to reuse job directory: $JOB" >&2; exit 2; }
mkdir -p "$JOB"
CONTROL=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_control_manifest"]["path"])' "$RUN")
LOG=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["stdout_log"])' "$RUN")
OWNER_OUTDIR=$("$PY" -c 'import json,sys; a=json.load(open(sys.argv[1]))["argv"]; xs=[x.split("=",1)[1] for x in a if x.startswith("NUMI_H706_FRACTIONAL_OUTPUT_DIR=")]; assert len(xs)==1; print(xs[0])' "$RUN")
EXECUTION=$RUN_DIR/execution.json
RUN_SHA=$(shasum -a 256 "$RUN" | awk '{print $1}')
LOG_SHA=$(shasum -a 256 "$LOG" | awk '{print $1}')
LABEL=$(basename "$RUN_DIR")
IDCFG=$JOB/identity-009.json
MASS=$JOB/unified-source-only-mass-013.json
CFG=$JOB/native-direct-O-009.json
ROLE_OUTDIR=$OWNER_OUTDIR/recorded-role-flow-1050

"$PY" "$IDENTITY" --run-manifest "$RUN" --source-control "$CONTROL" --source-closure "$CLOSURE" --build-manifest "$BUILD" --output-label "$LABEL-pre-mass-009" --output-config "$IDCFG" >"$JOB/identity-materialization.json"
"$PY" "$MASS_BUILDER" --run-manifest "$RUN" --identity-config "$IDCFG" --control-manifest "$CONTROL" --source-closure "$CLOSURE" --execution "$EXECUTION" --stdout-log "$LOG" --output "$MASS" --failure-report "$JOB/mass-013-failure.json" >"$JOB/mass-013.stdout" 2>"$JOB/mass-013.stderr"
MASS_SHA=$(shasum -a 256 "$MASS" | awk '{print $1}')

[[ ! -e "$ROLE_OUTDIR" ]] || { echo "refusing to reuse recorded-flow directory: $ROLE_OUTDIR" >&2; exit 2; }
mkdir "$ROLE_OUTDIR"
"$PY" "$CONVERT" --run-manifest "$RUN" --run-manifest-sha256 "$RUN_SHA" --admission-receipt "$MASS" --admission-sha256 "$MASS_SHA" --perf-log "$LOG" --perf-log-sha256 "$LOG_SHA" --converter-binary "$CONVERTER" --output-dir "$ROLE_OUTDIR" >"$JOB/converter.stdout" 2>"$JOB/converter.stderr"
"$PY" "$MATERIALIZE_RECORDED" --run-manifest "$RUN" --identity-config "$IDCFG" --mass-receipt "$MASS" --source-closure "$CLOSURE" --build-manifest "$BUILD" --output-label "$LABEL" --output-config "$CFG" --recorded-flow-dir "$ROLE_OUTDIR" >"$JOB/materialization-009.json"

"$PY" "$PREPARE" readiness --manifest "$CFG" --out "$JOB/readiness.json"
"$PY" "$PREPARE" plan-source --manifest "$CFG" --out "$JOB/source-plan.json"
if bash "$JOB/source-plan.stage-01.sh" >"$JOB/source.stdout" 2>"$JOB/source.stderr"; then SRC_RC=0; else SRC_RC=$?; fi
"$PY" "$PREPARE" record-source --manifest "$CFG" --plan "$JOB/source-plan.json" --stdout "$JOB/source.stdout" --stderr "$JOB/source.stderr" --exit-code "$SRC_RC" --out "$JOB/source-receipt.json"
"$PY" "$PREPARE" plan-measurement --manifest "$CFG" --source-plan "$JOB/source-plan.json" --source-receipt "$JOB/source-receipt.json" --steps 2 --out "$JOB/motion-plan.json"
if bash "$JOB/motion-plan.stage-02.sh" >"$JOB/motion.stdout" 2>"$JOB/motion.stderr"; then MOTION_RC=0; else MOTION_RC=$?; fi
"$PY" "$PREPARE" record-measurement --manifest "$CFG" --plan "$JOB/motion-plan.json" --stdout "$JOB/motion.stdout" --stderr "$JOB/motion.stderr" --exit-code "$MOTION_RC" --out "$JOB/motion-observer.json"
OUTROOT=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_root"])' "$CFG")
"$PY" "$MOTION_VERIFY" --checkpoint "$CHECKPOINT" --motion-dir "$OUTROOT/install-motion" --stdout "$JOB/motion.stdout" --stderr "$JOB/motion.stderr" --observer "$JOB/motion-observer.json" --exporter "$EXPORTER" --out "$JOB/motion-geometry-report.json"
