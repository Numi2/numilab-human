from pathlib import Path
import hashlib,json,sys,gc,time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0,'/Users/n/numi-human-anatomy-finish-1378/src')
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_intersections as ci
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276');D=A/'liver-island-native-027-002'
O=A/'liver-main-shell-native-1384-001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not O.exists();O.mkdir()
 r={'status':'running','scope':'All retained main liver14–21 native triangles across8 accepted poses; exactcoordinate quotient preserves geometry, fullselfpairaudit and closedorientedconnected checks. No hidden tolerances or generic acceptance of preexisting pairs.','helper_sha256':sha(__file__),'captures':[],'pins':{}}
 decl=json.loads((D/'baseline/run-declaration.json').read_text());ex=json.loads((D/'baseline/execution.json').read_text());assert ex['returncode']==0 and not ex['changed_inputs']
 for step in decl['capture_steps']:
  t=time.monotonic();p=D/'baseline/native-run/accepted-geometry'/f'step-{step}.mrvpack';r['pins'][str(p)]=sha(p)
  v,s,_=ca._pack_surfaces(p,{(51010,x) for x in range(14,22)})
  f=np.concatenate([s[(51010,x)]['faces'] for x in range(14,22)]);u,inv=np.unique(v[f].reshape((-1,3)),axis=0,return_inverse=True);wf=inv.reshape((-1,3))
  assert len(wf)==58552
  edges=np.concatenate([wf[:,[0,1]],wf[:,[1,2]],wf[:,[2,0]]]);eu,ei,ec=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
  orientation=np.bincount(ei,weights=np.where(edges[:,0]<edges[:,1],1,-1));graph=coo_matrix((np.ones(len(eu)),(eu[:,0],eu[:,1])),shape=(len(u),len(u))).tocsr()
  comps,_=connected_components(graph,directed=False)
  closed=bool((ec==2).all() and (orientation==0).all() and comps==1)
  tri=u[wf].astype(float);c=u.mean(axis=0);tri-=c;vol=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6
  records=ca._exact_surface_records(u,wf);audit=ci._audit_pair(records,records,same_surface=True)
  row={'step':step,'face_count':len(wf),'used_exact_xyz_vertex_count':len(u),'welded_components':int(comps),'boundary_edges':int((ec==1).sum()),'nonmanifold_edges':int((ec>2).sum()),'orientation_conflicts':int((orientation!=0).sum()),'closed_oriented_single_shell':closed,'signed_volume_ml':float(vol*1e6),'self_pair_count':audit['count'],'allowed_shared_vertex_edge_pairs':audit['allowed_shared_vertex_or_edge_pairs'],'seconds':time.monotonic()-t,'self_pair_table':audit['triangle_pairs']}
  r['captures'].append(row);(O/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k!='self_pair_table'}),flush=True)
  assert closed and vol>0
  if audit['count']:
   r['status']='failed_unexplained_main_self_pairs';(O/'report.json').write_text(json.dumps(r,indent=2)+'\n');return
  del v,s,f,u,inv,wf,edges,eu,ei,ec,tri,records,audit,graph;gc.collect()
 assert all(sha(p)==h for p,h in r['pins'].items())
 r['status']='pass';(O/'report.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
