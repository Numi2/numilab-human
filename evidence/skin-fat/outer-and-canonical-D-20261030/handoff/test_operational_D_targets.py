"""Real-asset preparation regression; no volume or native admission."""
from pathlib import Path
import copy,csv,hashlib,importlib.util,json,tempfile,time
import numpy as np
HERE=Path(__file__).parent
P=HERE.parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
v=load("operational_D_validation",HERE/"canonical_d_operational_targets.py")
h=load("operational_D_handoff",HERE/"prepare_tetgen_handoff_697_typedD.py")
b=P/"canonical-D-boundary-binding-696/run-010-combined-operational-reference-candidate"
proof_path=b/"candidate-preflight-receipt.json"
proof=json.loads(proof_path.read_text())
deep_path=Path(proof["candidate_path"])
surface=h.load_surface_217(h.load_owner(),deep_path,"D")
rows=h.read_pins(b/"operational-protected-149-targets.csv")
m=json.loads(Path(proof["construction_map_path"]).read_text())
tests=[];start=time.monotonic()
def passed(name):tests.append(name)
result=h.load_696_d_attachment_map(proof_path,deep_path,surface,rows)
assert len(result["attachment_routes"])==14
assert min(r["recomputed_full86_confidence_weight"] for r in result["attachment_routes"].values())>=.85
assert min(r["minimum_surface_altitude_m"] for r in result["attachment_routes"].values())>=50e-6
assert result["preserves_immutable_historical_rows"]
passed("real_D010_14_independently_recomputed_registrations")
surfaces={"D":surface}
for role in ("O","I"):
    head=[r for r in rows if r["role"]==role and r["mapping_kind"]=="head_ring_boundary_target"]
    points=np.asarray([[float(r["target_world_"+a+"_m"]) for a in "xyz"] for r in head],np.float32)
    faces=np.asarray([[0,i,i+1] for i in range(1,len(points)-1)],np.int64)
    surfaces[role]={"points":points,"faces":faces}
routes=h.route_pins(rows,surfaces,result)
assert len(routes["surface_routes"])==149 and routes["volume_targets"]==[]
changed=[r for r in routes["surface_routes"] if r.get("stable_attachment_id") in {11704,21455,39356,40968,46481}]
assert len(changed)==5 and all(r["target_world_m_f32"]!=r["historical_target_world_m_f32"] for r in changed)
passed("all149_routes_use_current_operational_targets_and_preserve_history")
with tempfile.TemporaryDirectory(prefix="D010-operational-regression-") as td:
    td=Path(td)
    def expect_reject(name,change_proof=None,change_map=None,change_rows=None,change_gate=None,change_surface=None):
        p=copy.deepcopy(proof);mm=copy.deepcopy(m);rr=copy.deepcopy(rows)
        ss=surface if change_surface is None else change_surface(surface)
        if change_rows:
            change_rows(rr)
            csvpath=td/(name+".csv")
            with csvpath.open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
            p["operational_protected149_path"]=str(csvpath);p["operational_protected149_sha256"]=v.sha(csvpath)
        if change_map:
            change_map(mm)
            mp=td/(name+"-map.json");mp.write_text(json.dumps(mm))
            p["construction_map_path"]=str(mp);p["construction_map_sha256"]=v.sha(mp)
        if change_gate:
            g=json.loads(Path(p["changed_face_exact_gate_path"]).read_text());change_gate(g)
            gp=td/(name+"-gate.json");gp.write_text(json.dumps(g))
            p["changed_face_exact_gate_path"]=str(gp);p["changed_face_exact_gate_sha256"]=v.sha(gp)
        if change_proof:change_proof(p)
        pp=td/(name+"-proof.json");pp.write_text(json.dumps(p))
        try:v.validate(pp,deep_path,ss,rr)
        except (ValueError,KeyError,IndexError):passed(name)
        else:raise AssertionError("accepted corrupt fixture: "+name)
    expect_reject("reject_candidate_hash_mismatch",change_proof=lambda p:p.update(candidate_sha256="0"*64))
    expect_reject("reject_physical_admission_claim",change_proof=lambda p:p.update(physical_admission=True))
    expect_reject("reject_nonzero_runtime",change_proof=lambda p:p.update(runtime=1))
    expect_reject("reject_rewritten_historical_target",change_rows=lambda rr:rr[90].update(target_world_x_m="123"))
    expect_reject("reject_operational_row_mismatch",change_rows=lambda rr:rr[90].update(operational_reference_surface_vertex_row="-1"))
    expect_reject("reject_duplicate_canonical_inventory",change_rows=lambda rr:rr[0].update(canonical_protected_row="1"))
    expect_reject("reject_unverified_gate_count",change_gate=lambda g:g["gates"][0].update(count=1))
    expect_reject("reject_incomplete_gate_inventory",change_gate=lambda g:g.update(gates=g["gates"][:2]))
    # Keep receipt hashes self-consistent so registration is independently checked.
    def wrong_weight(mm):mm["canonical14_source_registration"]["8467"]["recomputed_full86_confidence_weight"]=.99
    def repin_gate_to_map(g):
        path=td/"reject_forged_full86-map.json";g["target_map_sha256"]=v.sha(path)
    expect_reject("reject_forged_full86",change_map=wrong_weight,change_gate=repin_gate_to_map)
    expect_reject("reject_changed_existing_D_point",change_surface=lambda ss:{**ss,"points":np.asarray(ss["points"],np.float32)+np.float32(1e-4)})
receipt={"status":"PASS","tests":tests,"test_count":len(tests),"elapsed_s":time.monotonic()-start,
         "candidate_sha256":v.sha(deep_path),"proof_sha256":v.sha(proof_path),
         "validator_sha256":v.sha(HERE/"canonical_d_operational_targets.py"),
         "handoff_sha256":v.sha(HERE/"prepare_tetgen_handoff_697_typedD.py"),
         "physical_admission":False,"runtime":0,"accepted_steps":0}
(HERE/"test_operational_D_targets.receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
