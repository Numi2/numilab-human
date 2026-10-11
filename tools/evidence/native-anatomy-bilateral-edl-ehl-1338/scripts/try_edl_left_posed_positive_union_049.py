from pathlib import Path
from fractions import Fraction
import sys,json,hashlib,importlib.util,signal,time,traceback
import numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"edl-left-posed-positive-union-049";O.mkdir(exist_ok=False)
OWNER=Path("/Users/n/numi-human-touching-loop-arrangement-1290/src");sys.path.insert(0,str(OWNER))
from numilab_human import cardiac_cavity_partition as part,cardiac_cavity_intersections as ci,cardiac_cavity_geometry as cg
from numilab_human.physiology import canonical
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
N=A/"ehl-restored-pair-native-mtp-two-001";T=N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";CP=A/"edl-left-protected-interface-binding-034/stable-22-unadmitted-iteration-1.npz";RP=N/"baseline/native-run/accepted-geometry/step-8000.receipt.json"
pins={str(x):h.sha(x) for x in [Path(__file__),H,T,CP,RP,Path(part.__file__),Path(ci.__file__),Path(cg.__file__),OWNER/"numilab_human/cardiac_face_arrangement.py"]}
z=dict(np.load(CP));t=h.load_tissue(T);row=h.row_data(t,22);rc=json.loads(RP.read_text());pose={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc["accepted_body_poses"]}
world=h.forward(z["vertices6"][:,:3],z["binding_indices"],z["weights"],row,t,pose);f=z["faces"];v,inv=np.unique(world,axis=0,return_inverse=True);qf=inv[f];exact=[tuple(Fraction.from_float(float(c)) for c in p) for p in v]
before=ci._audit_pair(ci._records(exact,qf.tolist()),ci._records(exact,qf.tolist()),same_surface=True)
report={"scope":"Bounded offline diagnostic of existing strictly-positive winding owner at actual K2 step8000 forward geometry. Four remaining EDL overlaps are geometrically excised in posed space; any later source lift must preserve original pure proxies and pass source/all-pose/neighbor gates. No native or source admission from this operation.","pins":pins,"step":8000,"before":before,"complete":False}
def save():(O/"report.json").write_text(json.dumps(report,indent=2)+"\n")
save()
def alarm(*_):raise TimeoutError("180second bounded posed union")
signal.signal(signal.SIGALRM,alarm);signal.alarm(180);start=time.monotonic()
try:
 u=part.construct_positive_winding_self_union({"vertices":exact,"triangles":qf.tolist(),"source_sha256":pins[str(CP)]})
 doc={"schema":"HumanPack.cardiac-cavity-exact-self-union-result.v1","status":"offline_exact_rational_candidate","stable_id":22,"coordinate_frame":"registered_world_at_accepted_K2_pose","accepted_pose_receipt":str(RP),"accepted_pose_receipt_sha256":pins[str(RP)],"source_candidate":str(CP),"source_candidate_sha256":pins[str(CP)],"source_tissue_sha256":pins[str(T)],"construction":u,"ancestry_by_output_face":[{**r,"candidate_source_face_origin":int(z["face_origins"][r["source_face"]])} for r in u["output_face_ancestry"]]}
 dest=O/"stable-22-posed-exact-union.json";dest.write_bytes(canonical(part.encode_rational(doc))+b"\n")
 of=np.asarray(u["output_mesh"]["triangles"],np.uint32);op=np.asarray([[float(x) for x in r] for r in u["output_mesh"]["vertices_m"]],np.float32)
 rr,deg=h.exact_rows(op,of,ci);aa=ci._audit_pair(rr,rr,same_surface=True) if not deg else {"count":None}
 top=cg.analyze_topology(op.astype(float).tolist(),of.tolist())
 report.update(union_path=str(dest),union_sha256=h.sha(dest),posed_float32_self=aa,posed_degenerate_faces=deg,posed_topology={k:v for k,v in top.items() if not isinstance(v,(dict,list))},output_vertices=len(op),output_faces=len(of))
except Exception as e:report.update(error=str(e),traceback=traceback.format_exc())
signal.alarm(0);report.update(complete=True,inputs_unchanged=all(h.sha(p)==v for p,v in pins.items()),wall_seconds=time.monotonic()-start);save()
print(json.dumps({k:report.get(k) for k in ["complete","error","output_vertices","output_faces","posed_float32_self","wall_seconds"]}),flush=True)
