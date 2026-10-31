"""Actual closed-star controls for optional construction-only quality exchange."""
from pathlib import Path
import numpy as np,importlib.util,json,hashlib
H=Path(__file__).resolve().parent
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
A=mod('strict379_control464',H/'transfer_surface_contractions_indexed_379_root.py')
B=mod('exchange461_control464',H/'transfer_surface_contractions_quality_exchange_461_root.py')
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=np.asarray([[0,0,0],[.1,0,0],[0,1e-5,0],[0,0,.001],[1e-4,2e-6,.0002]],np.float32)
outer=np.array([[0,1,2,3]],np.int64);old=A.m.orient(p,np.c_[np.full(4,4),A.boundary(outer)])
v={'p':p,'t':old,'mat':np.full(4,15,np.int32),'look':{A.key(x):i for i,x in enumerate(p)},'map':np.arange(5),'active':np.ones(4,bool),'boundary_keys':{tuple(sorted(map(int,f))) for f in A.boundary(old)}}
op={'retained_input_point_row':1,'collapsed_input_point_row':4,'retired_parent_face_rows':[]};sf=np.empty((0,3),np.int64)
strict,why=A.try_op(v,p,sf,op);default,why2=B.try_op(v,p,sf,op);allowed,why3=B.try_op(v,p,sf,op,allow_quality_exchange=True)
assert strict is None and default is None and why==why2=='new_quality_failure'
assert allowed is not None and why3=='PASS' and allowed['new_retained_quality_failures']==1 and allowed['bad_before']==3 and allowed['bad_after']==1,(why3,allowed)
assert A.m.certify(p,old,allowed['new'][~allowed['deleted']])==(True,'PASS')
# The optional path still rejects a native-quality count increase.
badop={**op,'retired_parent_face_rows':[0]}
an,reason=B.try_op(v,p,np.array([[4,0,1]],np.int64),badop,allow_quality_exchange=True)
assert an is None and reason=='requires_DP_local_cavity_retriangulation',reason
vm=dict(v);vm['mat']=np.where(A.m.passes(*A.m.measurements(p,old)),15,16).astype(np.int32)
an,reason=B.try_op(vm,p,sf,op,allow_quality_exchange=True)
assert an is None and reason=='material_quality_count_increase',reason
assert np.array_equal(v['p'],p) and np.array_equal(v['t'],old) and v['active'].all()
out=H/'paired-I-quality-exchange-465-root/controls001';out.mkdir(parents=True,exist_ok=False)
r={'status':'PASS_CLOSED_STAR_CONSTRUCTION_QUALITY_EXCHANGE_CONTROLS','physical_admission':False,'native_steps':0,'controls':['old_strict_default_rejects_good_regression','optional_strict_net_material_quality_descent','exact_positive_oriented_boundary_volume_preserved','nonboundary_source_face_rejected','material_quality_count_increase_rejected','input_not_mutated'],'old_bad':allowed['bad_before'],'new_bad':allowed['bad_after'],'new_retained_quality_failures':1,'native_quality_still_fails':True,'producer_sha256':sha(__file__),'operator_sha256':sha(H/'transfer_surface_contractions_quality_exchange_461_root.py')}
np.savez_compressed(out/'control.npz',points_world_m=p,input_tetrahedra=old,output_tetrahedra=allowed['new'][~allowed['deleted']]);(out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
