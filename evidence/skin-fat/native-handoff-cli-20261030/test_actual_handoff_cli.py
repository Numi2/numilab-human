"""Actual CLI routing/capacity regression; not positive-core qualification."""
from pathlib import Path
import hashlib,importlib.util,json,subprocess,sys,tempfile
import numpy as np
P=Path(__file__).resolve().parent.parent
HERE=Path(__file__).parent
H=P/"paired-region-rebuild-667/tetgen-handoff-697"
D=P/"canonical-D-boundary-binding-696/run-010-combined-operational-reference-candidate"
CAP=P/"native-physical-parent-mass-330/native-source-identity-control-003"
source330=CAP/"compiled-parent-cell-mass.csv"
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
h=load("cli_owner_703",H/"prepare_tetgen_handoff_697_typedD.py")
owner=h.load_owner()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix="human-fullbody-cli-contract-") as td:
    td=Path(td);core=td/"unqualified-test-core.npz"
    # This fixture is only hashed in plan mode. No ready receipt exists and no
    # claim is made that it is a cavity core of the human.
    np.savez_compressed(core,points_world_m=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32),
                        faces=np.asarray([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],np.int64))
    outer=P/"reduced-surface-lineage-702/O081-lineage-prepared.npz"
    inner=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/dnew-vs-i296-inventory-001/i328-cavity-audit-001/fill-003/I-f32-conditioning-367/condition-288-pair-connected-remesh-001-run-002/result/candidate-f32-typed.npz"
    deep=D/"candidate-D-canonical14-reference-remap.npz"
    base=[sys.executable,str(H/"prepare_tetgen_handoff_697_typedD.py"),
          "--outer",str(outer),"--inner",str(inner),"--deep",str(deep),
          "--collar",str(HERE/"current-canonical-collar.npz"),
          "--wall-csv",str(P/"volume-worker/head-collar/proposal-001/donor-sidewall-faces.csv"),
          "--parent-cells",str(source330)+".tets.csv","--parent-nodes",str(source330)+".nodes.csv",
          "--protected-targets",str(D/"operational-protected-149-targets.csv"),
          "--canonical-D-proof",str(D/"candidate-preflight-receipt.json"),
          "--tetgen-binary",str(owner.DEFAULT_TETGEN),
          "--inner-hole-core",str(core),"--deep-hole-core",str(core),
          "--source330-base",str(source330),"--poses",str(CAP/"registered-skin-poses-100.json"),
          "--full86-weights",str(source330)+".weights.csv"]
    plan=td/"plan.json"
    cmd=base+["--output",str(td/"no-tetgen-output"),"--manifest-out",str(plan)]
    r=subprocess.run(cmd,text=True,capture_output=True)
    if r.returncode:raise RuntimeError(r.stderr[-3000:])
    obj=json.loads(plan.read_text())
    assert obj["status"]=="PLAN_ONLY_FAIL_NATIVE_CAPACITY"
    assert obj["native_capacity_preflight"]["status"]=="FAIL_NATIVE_CAPACITY"
    assert obj["protected149"]["count"]==len(obj["protected149"]["surface_routes"])==149
    assert obj["paired_rim"]=={"skin_faces":90,"fat_faces":90,"wall_rows":180,"single_I_input_domain":True}
    assert obj["surface_counts"]["O"]["faces"]==76937
    assert obj["surface_counts"]["I"]["faces"]==205239
    assert obj["surface_counts"]["D"]["faces"]==257469
    assert not (td/"no-tetgen-output").exists()
    assert not obj["native_runtime"] and not obj["physical_admission"]
    # The command must refuse execution before allocation or TetGen when the
    # necessary native capacity bound fails, regardless of optional arguments.
    exec_cmd=base+["--output",str(td/"no-execution-output"),"--manifest-out",str(td/"no-execution.json"),"--execute"]
    failure=subprocess.run(exec_cmd,text=True,capture_output=True)
    assert failure.returncode!=0 and "refusing impossible native-volume allocation" in failure.stderr
    assert not (td/"no-execution-output").exists()
    receipt={"status":"PASS_CLI_ROUTING_AND_CAPACITY_REJECTION_ONLY",
             "checks":["actual CLI routes prepared O, direct typed I and typed D","all149 current routes retained",
                       "paired90/90 rims reused","explicit plan-capacity failure status","execution refused before TetGen/allocation"],
             "surface_counts":obj["surface_counts"],"native_capacity_preflight":obj["native_capacity_preflight"],
             "handoff_sha256":sha(H/"prepare_tetgen_handoff_697_typedD.py"),
             "runner_sha256":sha(H/"run_native_domain_217_typedD_697.py"),
             "registered_surface_owner_sha256":sha(H/"registered_surface_handoff.py"),
             "collar_sha256":sha(HERE/"current-canonical-collar.npz"),
             "physical_admission":False,"runtime":0,"accepted_steps":0,"tetgen_run":False,
             "cavity_core_qualification":"NOT_RUN; deliberately unqualified fixture only hashed by plan mode"}
    (HERE/"test_actual_handoff_cli.receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))
