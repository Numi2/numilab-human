import hashlib,json,sys
from pathlib import Path
import numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
P=A/"back-parent-composition-014-007/patches/row-153.npz"
D=A/"back-forearm-native-014-015-back455-right177"
PACK=D/"baseline/native-run/rejected-surface-audit/step-0.mrvpack"
PREP=D/"preparation.json"
REST=A/"back-parent-composition-014-007/rest-preflight.json"
NATIVE=D/"baseline/native-run/native.log"
EXEC=D/"baseline/execution.json"
OUT=Path(__file__).parent/"report.json"
sys.path.insert(0,"/Users/n/numi-human-left-pq-publication-1344/src")
from numilab_human import common_atlas_skin_clearance as ca

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
patch=np.load(P); rest=np.asarray(patch["vertices6"][:,:3],dtype="<f4");faces=np.asarray(patch["faces"],dtype=np.int64)
world,surfs,info=ca._pack_surfaces(PACK,{(51005,153)})
wf=np.asarray(surfs[(51005,153)]["faces"],dtype=np.int64)
assert len(wf)==len(faces)
def area2(pos,tri):
 t=pos[tri].astype(np.float64);return np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
ra=area2(rest,faces);wa=area2(world,wf)
zeros=np.flatnonzero(wa==0)
fi=4040; rf=faces[fi];pf=wf[fi]
assert np.array_equal(world[pf[0]],world[pf[2]])
assert ra[fi]>0 and wa[fi]==0
rest_pre=json.loads(REST.read_text()); execution=json.loads(EXEC.read_text())
report={"schema":"numi.human.back455.step0.transform-degeneracy-audit.v1","gpu_used":False,"native_run_relaunched":False,
 "scene":str(D),"native_step0_surface_audit":"rejected","native_error":"whole native mesh contains invalid triangles: zero_area=1545 nonfinite_area=0 semantic=51005 stable_id=153 local_triangle=4040",
 "pack_inventory":info,"row153_face_count":int(len(faces)),"row153_rest_exact_f32_zero_area_faces":int(np.count_nonzero(ra==0)),
 "row153_capture_exact_f32_zero_area_faces":int(np.count_nonzero(wa==0)),"capture_zero_area_face_sample":zeros[:24].astype(int).tolist(),
 "native_reported_zero_area_faces":1545,"native_reported_nonfinite_area_faces":0,
 "face_4040":{"local_vertex_ids":rf.astype(int).tolist(),"pack_vertex_ids":pf.astype(int).tolist(),
  "rest_xyz_f32_m":rest[rf].tolist(),"captured_xyz_f32_m":world[pf].tolist(),
  "rest_doubled_area_m2":float(ra[fi]),"captured_doubled_area_m2":float(wa[fi]),
  "source_edge_y_delta_m":float(abs(rest[rf[0],1]-rest[rf[2],1])),
  "captured_edge_ulp_m_at_y":float(abs(np.spacing(np.float32(world[pf[0],1]))))},
 "row153_loader_rest_preflight_max_um":float(rest_pre["rows"]["153"]["max_um"]),
 "execution":{"changed_inputs":execution.get("changed_inputs"),"argv_matches":execution.get("native_argv_matches_prepared_cli_preview"),"environment_matches":execution.get("native_environment_matches_prepared_cli_preview")},
 "source_hashes":{"patch153_sha256":sha(P),"accepted_surface_rejection_pack_sha256":sha(PACK),"preparation_sha256":sha(PREP),"rest_preflight_sha256":sha(REST),"native_log_sha256":sha(NATIVE),"execution_receipt_sha256":sha(EXEC),"preparer_sha256":sha(A/"prepare_back_parent_composition_014_007.py")},
 "interpretation":"The source patch has no zero-area Float32 faces, and the whole-row 2um loader-rest gate passes. In the rejected step-0 native pack, at least 732 row153 triangles have exactly zero Float32 area; the reported triangle 4040 collapses because a 7.45nm source edge maps to identical captured Float32 endpoints near y=-0.607m (ULP about 59.6nm). The prep validates source-rest Float32 coordinates only; it does not apply its full runtime parent-term default-pose expansion to an exact face-area gate. The native audit reports 1,545 zero-area triangles, exceeding this direct exact-zero count, presumably due its near-zero area threshold; this report does not infer the threshold. No accepted run or admission claim."}
OUT.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
print(json.dumps({"path":str(OUT),"report":report},sort_keys=True))
