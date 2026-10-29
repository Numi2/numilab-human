from pathlib import Path
import importlib.util,tempfile,json,hashlib,argparse
import numpy as np
from scipy import sparse
P=Path(__file__).resolve().parent
def load(name,file):
 sp=importlib.util.spec_from_file_location(name,P/file);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
e=load("extract475","extract_contact_construction_constraints_475.py");m=load("fit476","fit_connected_measured_contact_476.py")
checks=[]
with tempfile.TemporaryDirectory(prefix="human-geometry-test477-",dir=P) as td:
 d=Path(td);base=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
 points=np.r_[base,base+np.array([3.,0.,0.])]
 np.savez_compressed(d/"candidate.npz",points_reference_world_m_f64=points,volume_tetrahedra=np.array([[0,1,2,3],[4,5,6,7]]),material_index=np.array([15,16]))
 faces=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
 def run_bank(shift):
  b=d/"bank.npz";np.savez_compressed(b,points_world_m_f64=base*.05+np.asarray(shift),faces_f64=faces)
  manifest=d/"manifest.json";manifest.write_text(json.dumps({"operands":[{"index":1,"body":1,"bone":2,"operand_path":str(b),"operand_sha256":hashlib.sha256(b.read_bytes()).hexdigest()}]}))
  return e.extract(d/"candidate.npz",manifest)
 ar,report=run_bank([.1,.1,.1])
 assert report["positive_pairs"]==1 and report["localized_positive_pairs"]==0 and len(report["unlocalized_containments_or_contact_rows"])==1
 assert ar["volume_node_rows"].shape==(0,3)
 rejected=False
 try:m.require_localized(report)
 except ValueError:rejected=True
 assert rejected;checks.append("contained bone without boundary witness is counted and rejected")
 ar,report=run_bank([10.,10.,10.])
 assert report["positive_pairs"]==0 and report["localized_positive_pairs"]==0 and not report["unlocalized_containments_or_contact_rows"]
 assert ar["volume_node_rows"].shape==(0,3);m.require_localized(report);checks.append("empty contact domain retains valid typed shapes")
m.self_test();checks.append("known constrained projection")
m.CONTACT_MATRIX=sparse.csc_matrix([[1.,1.]]);m.CONTACT_TARGET=np.array([3.])
x,r=m.solve_qp(sparse.eye(2,format="csc"),np.zeros(2),sparse.eye(2,format="csc"),np.zeros(2),np.ones(2))
assert np.allclose(x,[1.,1.],atol=1e-9) and abs(r["contact_slack_sum_m"]-1.)<1e-9
checks.append("soft contact cannot override hard capacity; unresolved slack retained")
m.CONTACT_MATRIX=None;m.CONTACT_TARGET=None
rejected=False
try:m.solve_qp(sparse.eye(1,format="csc"),np.zeros(1),sparse.csc_matrix([[1.],[1.]]),np.array([2.,-np.inf]),np.array([np.inf,1.]))
except RuntimeError:rejected=True
assert rejected;checks.append("infeasible hard construction constraints rejected")
print(json.dumps({"status":"passed","checks":checks}))
