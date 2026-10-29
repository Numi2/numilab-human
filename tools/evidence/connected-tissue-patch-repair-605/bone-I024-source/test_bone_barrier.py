from pathlib import Path
import importlib.util,sys,json,hashlib
import numpy as np
p=Path(__file__).resolve().parent;P=p.parent
def load(n,path):
 s=importlib.util.spec_from_file_location(n,path);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
h=load("bone609",p/"bone_barrier.py")
r=load("remesh609",P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py")
pred=r.load_exact_owner()
points=np.asarray([[0,0,0],[1,0,0],[0,1,0]],np.float32)
raw,_=pred.float32_surface_records(points.astype(float),[(0,1,2)],source_seam_groups=())
bone=h.reidentify_records(raw,vertex_offset=3)
q=raw[0];query=(q[0],q[1],q[2],100,q[4])
result=pred._audit_pair_prepared_first(pred._prepare_surface_aabb(bone),[query],same_surface=True)
assert result["count"]==1 and result["triangle_pairs"]==[[-1,100]],result
above,_=pred.float32_surface_records((points+np.asarray([0,0,1],np.float32)).astype(float),[(0,1,2)],source_seam_groups=())
a=above[0];a=(a[0],a[1],a[2],101,a[4])
assert pred._audit_pair_prepared_first(pred._prepare_surface_aabb(bone),[a],same_surface=True)["count"]==0
assert h.exact_common_lattice_key([.1,0,0])!=h.exact_common_lattice_key([float(np.float32(.1)),0,0])
assert h.exact_common_lattice_key([1.,0,0])==pred.float32_point_lattice_key([1.,0,0])
for point in ([float("nan"),0,0],[float.fromhex("0x0.0000000000001p-1022"),0,0]):
 try:h.exact_common_lattice_key(point)
 except ValueError:pass
 else:raise AssertionError("invalid common-lattice point accepted")
out={"schema":"existing-bone-record-remesh-barrier-regression.v1","status":"PASS","helper_sha256":h.sha(p/"bone_barrier.py"),"same_coordinate_distinct_structure_intersection_detected":True,"negative_face_ids_work_with_existing_prepared_predicate":True,"disjoint_structure_passes":True,"float64_operand_not_rounded_to_float32":True,"invalid_coordinates_rejected":True,"physical_admission":False}
(p/"regression.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out))
