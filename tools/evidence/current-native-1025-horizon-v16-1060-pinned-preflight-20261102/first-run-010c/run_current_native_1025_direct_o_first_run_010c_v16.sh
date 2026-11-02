#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 9 ]]; then
  echo "usage: $0 RUN_MANIFEST SOURCE_CLOSURE BUILD_MANIFEST CHECKPOINT ORIENTATION_AUDIT AUDIT_SHA256 ORIENTATION_OWNER OWNER_SHA256 JOB_DIR" >&2
  exit 64
fi
resolve_file() {
  if [[ "$1" = /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; fi
}
RUN=$(resolve_file "$1")
CLOSURE=$(resolve_file "$2")
BUILD=$(resolve_file "$3")
CHECKPOINT=$(resolve_file "$4")
AUDIT=$(resolve_file "$5")
AUDIT_SHA=$6
OWNER=$(resolve_file "$7")
OWNER_SHA=$8
JOB_INPUT=$9
if [[ "$JOB_INPUT" = /* ]]; then JOB=$JOB_INPUT; else JOB=$PWD/$JOB_INPUT; fi
PY=/Users/numi/numi-human-prep-venv-20261005/bin/python3
P=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-native-1025-launch/current-native-1025-direct-o-007-recorded-flow-scope
CONTRACT=$P/recorded_flow_contract_009_postpass.py
IDENTITY=$P/materialize_current_fractional_013_identity_009.py
MASS_BUILDER=$P/build_current_fractional_mass_receipt_013_nested_controls_compat_009.py
MATERIALIZE_RECORDED=$P/materialize_current_native_1025_direct_o_009_recorded_flow_dir.py
PREPARE=$P/prepare_current_native_1025_direct_o_009.py
CONVERT=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-fractional-recorded-flow-1050/solver-free-converter-001/convert_current_fractional_combined_flow.py
CONVERTER=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-fractional-recorded-flow-1050/solver-free-converter-001/convert-current-fractional-combined-flow
MOTION_VERIFY=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-motion-1026-verifier/verify_current_motion_geometry_1026_v16.py
HORIZON_PREP=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-native-1025-launch/current-native-1025-direct-o-010b-horizon-continuation/prepare_current_native_1025_horizon_010b.py
EXPORTER=/Users/numi/Work/skin-fat-native-integration-209/build-native/numi-human-accepted-skin-export
V16_SHA=35f18c19808bb4da21778f012d2054c12efcd744d3008857d9723f3b4acbea09
HORIZON_SHA=ddb19ecf5bffd79569cd888d92675f0207a9ba8680e905ba0a39f8d4a4599d00
for path in "$RUN" "$CLOSURE" "$BUILD" "$CHECKPOINT" "$AUDIT" "$OWNER" "$PY" "$CONTRACT" "$IDENTITY" "$MASS_BUILDER" "$MATERIALIZE_RECORDED" "$PREPARE" "$CONVERT" "$CONVERTER" "$MOTION_VERIFY" "$HORIZON_PREP" "$EXPORTER"; do
  [[ -f "$path" ]] || { echo "required input is missing: $path" >&2; exit 2; }
done
[[ "$AUDIT_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "orientation audit SHA must be 64 lowercase hex characters" >&2; exit 2; }
[[ "$OWNER_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "orientation owner SHA must be 64 lowercase hex characters" >&2; exit 2; }
ACTUAL_AUDIT_SHA=$(shasum -a 256 "$AUDIT" | awk '{print $1}')
ACTUAL_OWNER_SHA=$(shasum -a 256 "$OWNER" | awk '{print $1}')
[[ "$ACTUAL_AUDIT_SHA" == "$AUDIT_SHA" ]] || { echo "fresh orientation audit SHA mismatch" >&2; exit 2; }
[[ "$ACTUAL_OWNER_SHA" == "$OWNER_SHA" ]] || { echo "fresh orientation owner SHA mismatch" >&2; exit 2; }
[[ "$(shasum -a 256 "$MOTION_VERIFY" | awk '{print $1}')" == "$V16_SHA" ]] || { echo "v16 verifier SHA mismatch" >&2; exit 2; }
[[ "$(shasum -a 256 "$HORIZON_PREP" | awk '{print $1}')" == "$HORIZON_SHA" ]] || { echo "010b horizon verifier/adapter SHA mismatch" >&2; exit 2; }
[[ ! -e "$JOB" ]] || { echo "refusing to reuse job directory: $JOB" >&2; exit 2; }
"$PY" - "$CHECKPOINT" "$AUDIT" "$AUDIT_SHA" "$OWNER" "$OWNER_SHA" <<'PY'
import hashlib, json, pathlib, sys
checkpoint_path, audit_path, audit_sha, owner_path, owner_sha = sys.argv[1:]
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
ck=json.loads(pathlib.Path(checkpoint_path).read_text()); audit=json.loads(pathlib.Path(audit_path).read_text())
if audit.get("schema") != "numi.h706.1022-role-surface-orientation-audit.v1" or audit.get("status") != "PASS_EXACT_1022_SIGNED_ROLE_CONVENTIONS": raise SystemExit("orientation audit is not a passing 1022 source audit")
if audit.get("physical_admission") is not False or audit.get("native_steps") != 0: raise SystemExit("orientation audit overclaims physical/native evidence")
if audit.get("validator") != {"path": str(pathlib.Path(owner_path).resolve()), "sha256": owner_sha}: raise SystemExit("orientation audit validator path/SHA does not match supplied owner")
if sha(audit_path) != audit_sha or sha(owner_path) != owner_sha: raise SystemExit("orientation input hash changed during preflight")
state=audit.get("current_state",{}); ck_state=ck.get("source_state",{})
if not isinstance(ck_state,dict) or state.get("path") != ck_state.get("path") or state.get("sha256") != ck_state.get("sha256"): raise SystemExit("orientation audit current-state pin differs from the integration checkpoint")
receipt=audit.get("receipt",{})
rp=pathlib.Path(receipt.get("path",""))
if not rp.is_file() or sha(rp) != receipt.get("sha256"): raise SystemExit("fresh paired-shell receipt pin is missing or stale")
print("PASS_ORIENTATION_INPUT_PINS")
PY
mkdir -p "$JOB"
"$PY" "$MOTION_VERIFY" --input-preflight --checkpoint "$CHECKPOINT" --out "$JOB/orientation-input-preflight.json" --orientation-audit "$AUDIT" --orientation-audit-sha256 "$AUDIT_SHA" --orientation-owner "$OWNER" --orientation-owner-sha256 "$OWNER_SHA" >"$JOB/orientation-preflight.stdout" 2>"$JOB/orientation-preflight.stderr"
CONTROL=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_control_manifest"]["path"])' "$RUN")
LOG=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["stdout_log"])' "$RUN")
OWNER_OUTDIR=$("$PY" -c 'import json,sys; a=json.load(open(sys.argv[1]))["argv"]; xs=[x.split("=",1)[1] for x in a if x.startswith("NUMI_H706_FRACTIONAL_OUTPUT_DIR=")]; assert len(xs)==1; print(xs[0])' "$RUN")
RUN_DIR=$(dirname "$RUN")
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
MOTION_DIR=$OUTROOT/install-motion
"$PY" "$MOTION_VERIFY" --checkpoint "$CHECKPOINT" --motion-dir "$MOTION_DIR" --stdout "$JOB/motion.stdout" --stderr "$JOB/motion.stderr" --observer "$JOB/motion-observer.json" --exporter "$EXPORTER" --out "$JOB/motion-geometry-report.json" --orientation-audit "$AUDIT" --orientation-audit-sha256 "$AUDIT_SHA" --orientation-owner "$OWNER" --orientation-owner-sha256 "$OWNER_SHA" >"$JOB/geometry.stdout" 2>"$JOB/geometry.stderr"
"$PY" "$HORIZON_PREP" verify-stage --manifest "$CFG" --checkpoint "$CHECKPOINT" --source-plan "$JOB/source-plan.json" --source-receipt "$JOB/source-receipt.json" --motion-plan "$JOB/motion-plan.json" --observer "$JOB/motion-observer.json" --route "$JOB/motion-observer.route.json" --stdout "$JOB/motion.stdout" --stderr "$JOB/motion.stderr" --geometry-report "$JOB/motion-geometry-report.json" --verifier "$MOTION_VERIFY" --verifier-sha256 "$V16_SHA" --orientation-audit "$AUDIT" --orientation-audit-sha256 "$AUDIT_SHA" --orientation-owner "$OWNER" --orientation-owner-sha256 "$OWNER_SHA" --out "$JOB/first-horizon-gate.json" >"$JOB/first-horizon-gate.stdout" 2>"$JOB/first-horizon-gate.stderr"
cat "$JOB/first-horizon-gate.json"
