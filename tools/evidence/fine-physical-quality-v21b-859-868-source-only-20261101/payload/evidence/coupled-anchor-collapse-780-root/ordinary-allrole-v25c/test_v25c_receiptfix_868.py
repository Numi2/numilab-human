from pathlib import Path
import importlib.util, json, sys

HERE=Path(__file__).resolve(); H=HERE.parents[2]
owner_path=H/"continue_coupled_quality_points_v24c_nested_helper_receipt_868.py"
spec=importlib.util.spec_from_file_location("receiptfix_owner_868",owner_path)
owner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=owner;spec.loader.exec_module(owner)
root=H/"all-quality-866-rollback-root"; proposal=H/"all-quality-866-root"
receipt_path=root/"witness-component-rollback/rollback-preparation-receipt.json"
prep=json.loads(receipt_path.read_text())
paths={
 "source_baseline_state":H/"all-quality-859-qualified-root/quality-working-state.json",
 "state":proposal/"state.json",
 "config":proposal/"context-config.json",
 "failed_805_audit":H/"all-quality-866-audit805b-root/proposal-exact-805/report.json",
 "rollback_state":root/"witness-component-rollback/state.json",
 "rollback_config":root/"witness-component-rollback/context-config.json",
 "rollback_plan":root/"witness-component-rollback/rollback-plan.json",
}
owner._validate_rollback_preparation_receipt_pins(prep,paths["source_baseline_state"],paths["state"],paths["config"],paths["failed_805_audit"],paths["rollback_state"],paths["rollback_config"],paths["rollback_plan"])
assert "helper" not in prep and "helper" in prep["inputs"]
# Negative controls: no helper, conflicting locations, and wrong helper SHA all fail closed.
for mutate in ("missing","duplicate","wrong_sha"):
 bad=json.loads(json.dumps(prep))
 if mutate=="missing":del bad["inputs"]["helper"]
 elif mutate=="duplicate":bad["helper"]=bad["inputs"]["helper"]
 else:bad["inputs"]["helper"]["sha256"]="0"*64
 try:
  owner._validate_rollback_preparation_receipt_pins(bad,paths["source_baseline_state"],paths["state"],paths["config"],paths["failed_805_audit"],paths["rollback_state"],paths["rollback_config"],paths["rollback_plan"])
 except ValueError:pass
 else:raise AssertionError("receipt control unexpectedly accepted: "+mutate)
print("PASS actual 866 rollback receipt compatibility: nested exact helper pin and all top-level/input pins; 3 fail-closed mutation controls")
