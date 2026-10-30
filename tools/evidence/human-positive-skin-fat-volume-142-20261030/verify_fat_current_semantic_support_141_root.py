"""Independent same-material support consistency check; no physical admission."""
from pathlib import Path
import importlib.util,sys,json,hashlib,time
import numpy as np
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.monotonic();folder=H/'fat-current-semantic-support-141-root';rp=folder/'receipt.json';r=json.loads(rp.read_text());sp=Path(r['output']['path'])
 assert sha(sp)==r['output']['sha256']=='f3081e59fd88a88b48aa3d9caa8153609f6c24fcc0aac32d069c634ddad12b2b'
 assert r['physical_admission']is False and r['native_steps']==0
 with np.load(sp)as z:a={k:z[k]for k in z.files}
 vol=H/'fat-positive-PLC-prefix-compaction-135-root/fat-positive-compact-PLC-prefix.npz'
 assert sha(vol)==str(a['volume_sha256'].item())
 with np.load(vol)as z:p=z['points_world_m'];t=z['tetrahedra']
 sys.path.insert(0,str(H/'downstream-003-current-typed-228/native-domain-217-source-005'))
 op=H/'current-semantic-atlas-088/export_authoring_registered_surface_sidecars_230_semantic_088.py'
 assert sha(op)==r['owner']['sha256']
 spec=importlib.util.spec_from_file_location('verify141reg',op);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 base=H.parent/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv'
 parent,_,_,pm=m.load_parent_current(base)
 assert pm==r['source330'];g=a['group_tuples'];n=len(p);c=len(t)
 assert g.shape==(314,3)and np.all(g[:,0]==16)and len(set(map(tuple,g)))==314
 assert set(map(tuple,g))=={tuple(int(parent[k][v])for k in ('materials','donors','regions'))for v in np.flatnonzero(parent['materials']==16)}
 def csr(off,vals,count,limit):
  assert off.shape==(count+1,)and off[0]==0 and off[-1]==len(vals)and np.all(np.diff(off)>0)
  assert vals.min()>=0 and vals.max()<limit
 csr(a['node_group_offsets'],a['node_group_ids'],n,len(g))
 csr(a['node_group_parent_tet_offsets'],a['node_group_parent_tet_rows'],len(a['node_group_ids']),len(parent['tets']))
 csr(a['cell_group_offsets'],a['cell_group_ids'],c,len(g))
 csr(a['cell_group_parent_tet_offsets'],a['cell_group_parent_tet_rows'],len(a['cell_group_ids']),len(parent['tets']))
 for prefix in ('node','cell'):
  ids=a[prefix+'_group_ids'];off=a[prefix+'_group_parent_tet_offsets'];rows=a[prefix+'_group_parent_tet_rows']
  repeated=np.repeat(ids,np.diff(off));labels=np.stack([parent[k][rows]for k in ('materials','donors','regions')],axis=1)
  assert np.array_equal(labels,g[repeated])
 scores=a['cell_group_p1_score_numerators']/a['cell_group_p1_score_denominators']
 sums=np.add.reduceat(scores,a['cell_group_offsets'][:-1])
 assert np.max(np.abs(sums-1))<1e-14
 residual_max=0.
 for off in range(0,n,65536):
  rows=a['node_parent_chosen_tet_row'][off:off+65536];b=a['node_parent_barycentric'][off:off+65536]
  assert np.all(parent['materials'][rows]==16)and b.min()>=-1e-8 and np.max(np.abs(b.sum(axis=1)-1))<1e-8
  q=np.einsum('ij,ijk->ik',b,parent['points'][parent['tets'][rows]])+a['node_parent_residual_world_m'][off:off+65536]
  err=np.linalg.norm(q-p[off:off+65536].astype(float),axis=1)
  residual_max=max(residual_max,float(err.max()))
 assert residual_max<1e-10
 # Deterministic independent exact-rational P1/full-parent support checks.
 sample=np.unique(np.concatenate((np.arange(min(c,32)),np.linspace(0,c-1,2048,dtype=int))))
 from fractions import Fraction
 for row in sample:
  expected={};parents={}
  for v in t[row]:
   l,u=a['node_group_offsets'][v:v+2];ids=a['node_group_ids'][l:u]
   for slot,gid in zip(range(l,u),ids):
    expected[int(gid)]=expected.get(int(gid),Fraction())+Fraction(1,4*len(ids))
    lo,hi=a['node_group_parent_tet_offsets'][slot:slot+2]
    parents.setdefault(int(gid),set()).update(a['node_group_parent_tet_rows'][lo:hi].tolist())
  lo,hi=a['cell_group_offsets'][row:row+2]
  assert set(a['cell_group_ids'][lo:hi])==set(expected)
  for slot in range(lo,hi):
   gid=int(a['cell_group_ids'][slot]);assert Fraction(int(a['cell_group_p1_score_numerators'][slot]),int(a['cell_group_p1_score_denominators'][slot]))==expected[gid]
   x,y=a['cell_group_parent_tet_offsets'][slot:slot+2];assert set(a['cell_group_parent_tet_rows'][x:y])==parents[gid]
 report={'status':'PASS_INDEPENDENT_CURRENT_FAT_SUPPORT_CONSISTENCY','physical_admission':False,'native_steps':0,'mass_transport':'NOT_RUN','support_sha256':sha(sp),'producer_receipt_sha256':sha(rp),'nodes':n,'cells':c,'source_groups':314,'cell_group_slots':len(scores),'maximum_P1_sum_error':float(np.max(np.abs(sums-1))),'maximum_parent_reconstruction_error_m':residual_max,'exact_rational_P1_parent_union_sampled_cells':len(sample),'full_checks':['All314same-material source groups','Every node/cell source-parent group identity','Every cell P1 partition of unity','Every chosen parent barycentric+residual reconstruction'],'elapsed_s':time.monotonic()-start}
 out=folder/'independent-consistency.json';out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
