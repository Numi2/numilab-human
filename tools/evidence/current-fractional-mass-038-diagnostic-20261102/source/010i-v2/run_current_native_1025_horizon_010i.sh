#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 15 ]]; then
  echo "usage: $0 MANIFEST CHECKPOINT SOURCE_PLAN SOURCE_RECEIPT PREVIOUS_GATE HORIZON_SPEC JOB_DIR V17C_VERIFIER V17C_SHA256 ORIENTATION_AUDIT AUDIT_SHA256 ORIENTATION_OWNER OWNER_SHA256 EXPORTER EXPORTER_SHA256" >&2
  echo "HORIZON_SPEC is steps:16|128|1024 or wall:SECONDS; each run is a fresh initialization after the prior gate passes." >&2
  exit 64
fi
resolve_file() {
  if [[ "$1" = /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; fi
}
MANIFEST=$(resolve_file "$1")
CHECKPOINT=$(resolve_file "$2")
SOURCE_PLAN=$(resolve_file "$3")
SOURCE_RECEIPT=$(resolve_file "$4")
PREVIOUS_GATE=$(resolve_file "$5")
SPEC=$6
JOB_INPUT=$7
VERIFIER=$(resolve_file "$8")
VERIFIER_SHA=$9
ORIENTATION_AUDIT=$(resolve_file "${10}")
ORIENTATION_AUDIT_SHA=${11}
ORIENTATION_OWNER=$(resolve_file "${12}")
ORIENTATION_OWNER_SHA=${13}
EXPORTER=$(resolve_file "${14}")
EXPORTER_SHA=${15}
PY=/Users/numi/numi-human-prep-venv-20261005/bin/python
PREP=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-native-1025-launch/current-native-1025-direct-o-010i-envelope-bound-v2/prepare_current_native_1025_horizon_010i.py
mkdir -p "$(dirname "$JOB_INPUT")"
JOB=$(resolve_file "$JOB_INPUT")
if [[ -e "$JOB" ]]; then echo "refusing to reuse continuation job directory: $JOB" >&2; exit 2; fi
CAPTURE_PROFILE=${NUMI_HUMAN_CAPTURE_PROFILE:-sampled-three}
case "$CAPTURE_PROFILE" in sampled-three|quartiles-six) ;; *) echo "NUMI_HUMAN_CAPTURE_PROFILE must be sampled-three or quartiles-six" >&2; exit 64 ;; esac
mkdir -p "$JOB"
PLAN=$JOB/horizon-plan.json
OBSERVER=$JOB/motion-observer.json
ROUTE=$JOB/motion-observer.route.json
REPORT=$JOB/motion-geometry-report.json
GATE=$JOB/horizon-gate.json
COMMON=(--manifest "$MANIFEST" --checkpoint "$CHECKPOINT" --source-plan "$SOURCE_PLAN" --source-receipt "$SOURCE_RECEIPT" --previous-gate "$PREVIOUS_GATE" --verifier "$VERIFIER" --verifier-sha256 "$VERIFIER_SHA" --orientation-audit "$ORIENTATION_AUDIT" --orientation-audit-sha256 "$ORIENTATION_AUDIT_SHA" --orientation-owner "$ORIENTATION_OWNER" --orientation-owner-sha256 "$ORIENTATION_OWNER_SHA" --exporter "$EXPORTER" --exporter-sha256 "$EXPORTER_SHA")
if [[ "$SPEC" == steps:* ]]; then
  STEPS=${SPEC#steps:}
  "$PY" "$PREP" plan-horizon "${COMMON[@]}" --steps "$STEPS" --capture-profile "$CAPTURE_PROFILE" --out "$PLAN" >"$JOB/plan.stdout" 2>"$JOB/plan.stderr"
elif [[ "$SPEC" == wall:* ]]; then
  WALL=${SPEC#wall:}
  "$PY" "$PREP" plan-duration "${COMMON[@]}" --target-wall-seconds "$WALL" --capture-profile "$CAPTURE_PROFILE" --out "$PLAN" >"$JOB/plan.stdout" 2>"$JOB/plan.stderr"
else
  echo "invalid HORIZON_SPEC" >&2; exit 64
fi
MOTION_DIR=$("$PY" - "$PLAN" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["stage"]["argv"][4])
PY
)
if bash "$JOB/horizon-plan.stage.sh" >"$JOB/motion.stdout" 2>"$JOB/motion.stderr"; then RUN_RC=0; else RUN_RC=$?; fi
"$PY" "$PREP" record --manifest "$MANIFEST" --source-plan "$SOURCE_PLAN" --source-receipt "$SOURCE_RECEIPT" --motion-plan "$PLAN" --stdout "$JOB/motion.stdout" --stderr "$JOB/motion.stderr" --exit-code "$RUN_RC" --observer-out "$OBSERVER" >"$JOB/record.stdout" 2>"$JOB/record.stderr"
"$PY" "$VERIFIER" --checkpoint "$CHECKPOINT" --motion-dir "$MOTION_DIR" --stdout "$JOB/motion.stdout" --stderr "$JOB/motion.stderr" --observer "$OBSERVER" --exporter "$EXPORTER" --out "$REPORT" --orientation-audit "$ORIENTATION_AUDIT" --orientation-audit-sha256 "$ORIENTATION_AUDIT_SHA" --orientation-owner "$ORIENTATION_OWNER" --orientation-owner-sha256 "$ORIENTATION_OWNER_SHA" >"$JOB/geometry.stdout" 2>"$JOB/geometry.stderr"
"$PY" "$PREP" verify-stage --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" --source-plan "$SOURCE_PLAN" --source-receipt "$SOURCE_RECEIPT" --motion-plan "$PLAN" --observer "$OBSERVER" --route "$ROUTE" --stdout "$JOB/motion.stdout" --stderr "$JOB/motion.stderr" --geometry-report "$REPORT" --verifier "$VERIFIER" --verifier-sha256 "$VERIFIER_SHA" --orientation-audit "$ORIENTATION_AUDIT" --orientation-audit-sha256 "$ORIENTATION_AUDIT_SHA" --orientation-owner "$ORIENTATION_OWNER" --orientation-owner-sha256 "$ORIENTATION_OWNER_SHA" --out "$GATE" >"$JOB/gate.stdout" 2>"$JOB/gate.stderr"
cat "$GATE"
