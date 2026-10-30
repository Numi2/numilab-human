"""Owner dependency and candidate-hash readiness contract; no physical admission."""
from pathlib import Path
import importlib.util,json,tempfile
H=Path(__file__).resolve().parent.parent/"paired-region-rebuild-667/tetgen-handoff-697"
spec=importlib.util.spec_from_file_location("handoff_readiness_test",H/"prepare_tetgen_handoff_697_typedD.py")
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
names=["outer","inner","deep","collar","wall_csv","protected_targets","canonical_d_proof","owner_runner",
       "handoff_script","capacity_owner","operational_target_owner","registered_surface_owner",
       "inner_hole_core_npz","deep_hole_core_npz"]
hashes={name:h.sha(H/"registered_surface_handoff.py") if name=="registered_surface_owner" else name+"-fixture" for name in names}
base=dict(schema=h.READY_SCHEMA,status="PASS",physical_admission=False,
          gates={k:"PASS" for k in h.GATES},inputs={n:{"sha256":v} for n,v in hashes.items()})
checks=[]
with tempfile.TemporaryDirectory() as td:
    path=Path(td)/"ready.json"
    path.write_text(json.dumps(base))
    assert h.validate_readiness(path,hashes)["status"]=="PASS";checks.append("all owner dependencies matched")
    for name in names:
        for mode in ("missing","changed"):
            obj=json.loads(json.dumps(base))
            if mode=="missing": del obj["inputs"][name]
            else: obj["inputs"][name]["sha256"]="wrong"
            path.write_text(json.dumps(obj))
            try:h.validate_readiness(path,hashes)
            except ValueError as e:assert name in str(e)
            else:raise AssertionError((name,mode))
            checks.append(mode+":"+name)
receipt=dict(status="PASS",test_count=len(checks),checks=checks,
             registered_surface_owner_sha256=h.sha(H/"registered_surface_handoff.py"),
             handoff_sha256=h.sha(H/"prepare_tetgen_handoff_697_typedD.py"),
             physical_admission=False,runtime=0,accepted_steps=0)
out=Path(__file__).with_suffix(".receipt.json");out.write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
