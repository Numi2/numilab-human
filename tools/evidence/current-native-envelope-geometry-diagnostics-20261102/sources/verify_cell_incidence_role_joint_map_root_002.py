from pathlib import Path
import numpy as np,json,hashlib,time
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
D=H/'joint-local-i-patch-preflight-002/cell-incidence-role-joint-map-root-002';D.mkdir(exist_ok=False);start=time.monotonic()
def pin(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
tp=H/'current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz'
assert pin(tp)['sha256']=='5d31c141b66628801a711c2e745f744f745611882a269e173eddfcd0e99d3372'
with np.load(tp) as z:p=z['points_reference_world_m_f64'].astype(np.float32);t=z['volume_tetrahedra'];mat=z['material_index'];sr={k:z['current_typed_role_source_row_to_joint_node_'+k] for k in ['O','I','D']};ar={k:z['current_typed_role_active_source_vertex_rows_'+k] for k in ['O','I','D']}
inputs={};maps={};res=[]
for role,mt,expected in [('Skin',15,'41291a7298517ab250a64e4bbda6bbee7591b7d2b594aacc1dce6d6fc78025ba'),('Fat',16,'b83df0f4a4ff183bd147f0c0d6236a22e3ea6ce1ae9e9531ffc1c457c2ffac88')]:
 path=H/('all-quality-1009-point-update-1018-root/actual-001/'+role.lower()+'-volume-point-updated.npz');inputs[role]=pin(path);assert inputs[role]['sha256']==expected
 with np.load(path) as z:v=z['points_world_m_f32'];tt=z['tetrahedra']
 ct=t[mat==mt];assert ct.shape==tt.shape
 mmin=np.full(len(v),len(p),np.int64);mmax=np.full(len(v),-1,np.int64)
 for a in range(0,len(tt),65536):
  rt=tt[a:a+65536];jt=ct[a:a+65536];rv=v[rt];jv=p[jt]
  same=(rv[:,:,None,:].view(np.uint32)==jv[:,None,:,:].view(np.uint32)).all(axis=3)
  assert np.all(same.sum(axis=2)==1) and np.all(same.sum(axis=1)==1),(role,a,'per-cell F32 simplex mismatch')
  chosen=np.take_along_axis(jt,np.argmax(same,axis=2),axis=1)
  np.minimum.at(mmin,rt.reshape(-1),chosen.reshape(-1));np.maximum.at(mmax,rt.reshape(-1),chosen.reshape(-1))
 active=np.unique(tt);assert np.array_equal(mmin[active],mmax[active]),(role,'inconsistent all-incidence identity')
 m=mmax;assert np.array_equal(v[active].view(np.uint32),p[m[active]].view(np.uint32))
 assert np.array_equal(np.sort(m[tt],axis=1),np.sort(ct,axis=1))
 maps[role+'_local_to_joint']=m
 res.append({'role':role,'allocated_nodes':len(v),'active_nodes':len(active),'unreferenced_nodes':int(np.sum(m<0)),'cell_rows':len(tt),'every_incident_cell_F32_simplex_exact':True,'every_active_vertex_consistent_joint_identity':True,'all_mapped_cell_simplex_rows_exact':True})
boundary=np.unique(np.concatenate([sr[k][ar[k]] for k in sr]));free=np.setdiff1d(np.arange(len(p)),boundary);maps.update({'O_I_D_boundary_joint_nodes':boundary,'free_joint_nodes':free});np.savez_compressed(D/'maps.npz',**maps)
out={'schema':'numi.h706.current1020.direct027-cut-volume-sensitivity-and-bounded-I-step.v1','status':'PASS_ALL_CELL_INCIDENCE_ROLE_JOINT_CROSSWALK','physical_admission':False,'native_steps':0,'source':pin(Path(__file__)),'typed':pin(tp),'role_inputs':inputs,'roles':res,'boundary_joint_node_count':len(boundary),'free_joint_node_count':len(free),'maps':pin(D/'maps.npz'),'elapsed_s':time.monotonic()-start,'method':'Each pinned role cell matches the same material-local pinned 433 cell row by unique exact F32 corner coordinates. A local vertex must select one identical joint vertex over every incident cell. Inactive allocation rows stay -1. No global coordinate nearest/alias inference and no legacy TetGen ID treated as NPZ index.'};(D/'report.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
