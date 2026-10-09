from pathlib import Path
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"): os.environ[k]="1"
import numpy as np,json,hashlib,sys,importlib.util,time
B=Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218")
O=B/"local-self-first-pose-001"
def j(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def imp(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
a=imp(B/"heldout-001/audit_six_holdouts_1218.py","audit")
fm=imp(a.FORWARD,"forward")
from numilab_human import common_atlas_skin_clearance as c,cardiac_cavity_intersections as ci
report005=B/"local-self-reduction-005/feasibility.json";rr=j(report005)
assert rr["solver_success"] and rr["dependency_complete_replay"]["array_bit_exact"]
tracked=rr["inputs_after"].copy()
for p,d in tracked.items():assert sha(p)==d
candidate_path=B/"local-self-reduction-004/unadmitted-source-proposal.npy"
candidate=np.load(candidate_path,allow_pickle=False);skin=a.load_skin(a.SKIN);ref=np.unique(skin["faces"]);faces=np.searchsorted(ref,skin["faces"])
pack=a.RUN/"accepted-geometry/step-4991.mrvpack";receipt=pack.with_suffix(".receipt.json");rd=j(receipt)
assert rd["pack_file_sha256"]==sha(pack)
pos,surfs,counts=c._pack_surfaces(pack,{(51007,1)});sf=surfs[(51007,1)]["faces"];base=int(sf.min())
assert np.array_equal(sf-base,skin["faces"])
cap=pos[base:base+skin["nv"]][ref].copy()[None,...];del pos,surfs
initial=j(a.RUN/"accepted-geometry/step-0.receipt.json")
for p in [Path(__file__),pack,receipt,report005,candidate_path,a.RUN/"accepted-geometry/step-0.receipt.json"]:tracked[str(p)]=sha(p)
fw,mr,selectors=fm.build_forward(skin=skin,source_positions=skin["pos"],referenced_ids=ref,captured_by_pose=cap,state_receipts=[{"step":4991,"body_poses":rd["accepted_registered_body_poses"],"respiratory_motion":rd["accepted_respiratory_motion"]}],initial_body_poses=initial["initial_anatomical_registration"]["body_poses"],map_receipt=initial,anatomy_parameters_path=initial["skin_source_mapping"]["anatomy_parameters"]["path"],respiration_source=Path("/Users/n/numi-human-resting-resp-source-20261006"),clearance_module=c)
t=time.monotonic();out=fw(candidate);assert out["diagnostics"]["admissible"]
world=np.asarray(out["world_positions_by_pose"],dtype="<f4")[0]
records=c._exact_surface_records(world,faces);degenerate=[] # owner raises on an exactly degenerate triangle
check=ci._audit_pair(records,records,same_surface=True)
after={p:sha(p) for p in tracked};assert after==tracked
result={"step":4991,"skin_self_intersection_count":int(check["count"]),"degenerate_faces":[int(x) for x in degenerate],"self_face_pairs":check["triangle_pairs"],"candidate_npy_sha256":sha(candidate_path),"dependency_replay_sha256":sha(report005),"inputs_before":tracked,"inputs_after":after,"inputs_unchanged":True,"elapsed_forward_and_self_audit_seconds":time.monotonic()-t,"qualification":"First failing pose quick exact self check only; no target audit, no asset or native admission"}
(O/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print({k:v for k,v in result.items() if not k.startswith("inputs")})
