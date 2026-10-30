import importlib.util, json, pathlib, hashlib
import numpy as np
p=pathlib.Path(__file__).parent
spec=importlib.util.spec_from_file_location("capacity",p/"native_capacity_preflight.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
tet=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],np.int64)
r=m.closed_shell_bound(tet,label="tet")
assert (r["boundary_euler_characteristic"],r["necessary_minimum_cells"])==(2,1)
two=np.concatenate((tet,tet+4))
r=m.closed_shell_bound(two,label="two")
assert (r["boundary_euler_characteristic"],r["necessary_minimum_cells"])==(4,2)
# Closed orientable genus-one 3x3 periodic triangular torus.
idx=lambda i,j:(i%3)*3+(j%3)
torus=[]
for i in range(3):
 for j in range(3):
  a,b,c,d=idx(i,j),idx(i+1,j),idx(i+1,j+1),idx(i,j+1)
  torus += [[a,b,c],[a,c,d]]
r=m.closed_shell_bound(torus,label="torus")
assert (r["boundary_euler_characteristic"],r["necessary_minimum_cells"])==(0,9)
for bad in (tet[:3], np.concatenate((tet,tet[:1])), tet[:,[0,0,2]],
            np.concatenate((tet[:3],tet[3:,[0,2,1]]))):
 try: m.closed_shell_bound(bad,label="invalid")
 except ValueError: pass
 else: raise AssertionError("invalid shell accepted")
surfaces={k:{"faces":tet} for k in ("O","I","D")}
r=m.volume_capacity_report(tet,tet,surfaces,max_nodes=12,max_cells=2)
assert r["status"]=="PASS_NECESSARY_BOUNDS_ONLY"
assert r["necessary_combined_nodes"]==12
assert r["necessary_combined_cells"]==2
assert m.volume_capacity_report(tet,tet,surfaces,max_nodes=11)["status"]=="FAIL_NATIVE_CAPACITY"
assert m.volume_capacity_report(tet,tet,surfaces,max_cells=1)["status"]=="FAIL_NATIVE_CAPACITY"
# Exercise the actual handoff adapter with open O/I/D triangular disks and
# exact fixed six-face collars. Each material closes into a triangular prism.
spec=importlib.util.spec_from_file_location("handoff_capacity",p/"prepare_tetgen_handoff_697_typedD.py")
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
owner=h.load_owner()
tri=np.array([[0,1,2]],np.int64)
pts=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],np.float32)
layers={role:{"points":pts+np.array([0,0,z],np.float32),"faces":tri}
        for role,z in (("O",0),("I",1),("D",2))}
wall=np.array([[0,1,4],[0,4,3],[1,2,5],[1,5,4],[2,0,3],[2,3,5]],np.int64)
adapter=h.volume_capacity_preflight(owner,layers,wall,wall)
assert adapter["necessary_combined_cells"]==6
assert adapter["necessary_combined_nodes"]==9
assert adapter["status"]=="PASS_NECESSARY_BOUNDS_ONLY"
out={"status":"PASS","physical_admission":False,"runtime":0,
     "tests":["tetrahedron","disconnected components","genus-one shell",
              "open/duplicate/degenerate/orientation rejection","node and cell capacity rejection","actual handoff collar orientation and shared-I node count"],
     "source_sha256":hashlib.sha256((p/"native_capacity_preflight.py").read_bytes()).hexdigest()}
(p/"test_native_capacity_preflight.receipt.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out))
