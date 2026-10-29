from pathlib import Path
import sys,hashlib,json,time,collections
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
sys.path.insert(0,str(P.parent/'authoring-deps'))
import igl
A=P/'reduced-physical-boundary-400/attempt-044-minimal-coupled-bone32-buffer'
def main():
 out=A/'whole-O-fine-projection-011'
 if out.exists():raise FileExistsError(out)
 base=P/'reduced-physical-boundary-400/attempt-030-connected-constrained-edge-fit/run-006/candidate.npz'
 fine=P/'surface-transfer-checkpoint-392/payload/O/candidate.npz'
 w=A/'whole-O-native-all185-010/witnesses.npz'
 assert hashlib.sha256(base.read_bytes()).hexdigest()=='888eb12450d0e69fb0ad3f5998159ab7a0a20d3e70b13c42ba416c61cbbede63'
 assert hashlib.sha256(fine.read_bytes()).hexdigest()=='2958a704db4b8b68422640a5e991c6b51af25f19d525c49adc8a742f0c207d7a'
 b=np.load(base);f=np.load(fine);z=np.load(w)
 faces=z['outer_source_faces'];ids=np.unique(faces);points=b['points_current_world_m_f32'].astype(np.float64)
 st=time.time();d2,fi,cp=igl.point_mesh_squared_distance(points[ids],f['points_world_m'].astype(np.float64),f['faces'])
 tri=f['points_world_m'].astype(np.float64)[f['faces'][fi]]
 e0=tri[:,1]-tri[:,0];e1=tri[:,2]-tri[:,0];v=cp-tri[:,0]
 g00=np.einsum('ij,ij->i',e0,e0);g01=np.einsum('ij,ij->i',e0,e1);g11=np.einsum('ij,ij->i',e1,e1)
 a=np.einsum('ij,ij->i',v,e0);c=np.einsum('ij,ij->i',v,e1);den=g00*g11-g01*g01
 if np.any(den<=0):raise ValueError('degenerate selected fine source face')
 u=(a*g11-c*g01)/den;vv=(c*g00-a*g01)/den;bary=np.c_[1-u-vv,u,vv]
 assert np.max(np.linalg.norm(np.einsum('ij,ijk->ik',bary,tri)-cp,axis=1))<1e-12
 wf=np.unique(z['triangle_pairs'][:,0]);wn=np.unique(faces[wf]);sel=np.isin(ids,wn)
 inc=collections.defaultdict(list)
 for k,row in enumerate(faces[wf]):
  for i,j in ((0,1),(1,2),(2,0)):inc[tuple(sorted((int(row[i]),int(row[j]))))].append(k)
 edges=[(i,j) for rows in inc.values() for i in rows for j in rows]
 mat=coo_matrix((np.ones(len(edges)),tuple(zip(*edges))),shape=(len(wf),len(wf))).tocsr()
 nc,lab=connected_components(mat)
 def stats(x):
  return {'count':len(x),'rms_m':float(np.sqrt(np.mean(x*x))),'max_m':float(np.max(x)),'p50_p95_p99_m':np.percentile(x,[50,95,99]).tolist(),'over_0p4mm':int((x>.0004).sum()),'over_2p5mm':int((x>.0025).sum())}
 out.mkdir()
 np.savez_compressed(out/'projection.npz',coarse_O_global_node_rows=ids,accepted_fine_O_face_rows=fi,accepted_fine_O_barycentric=bary,accepted_fine_O_projected_points_f64=cp,distance_m=np.sqrt(d2),witness_outer_face_rows=wf,witness_global_node_rows=wn,component_by_witness_outer_face_row=lab,outer_source_faces=faces)
 report={'status':'SOURCE_ONLY_COARSE_O_ACCEPTED_FINE_O_PROJECTION','all_O_nodes':stats(np.sqrt(d2)),'witness_O_nodes':stats(np.sqrt(d2[sel])),'witness_faces':len(wf),'components':nc,'component_sizes':sorted(np.bincount(lab).tolist(),reverse=True),'seconds':time.time()-st,'runtime_created':0,'accepted_steps':0,'physical_admission':False,'limits':['Nearest-point mapping is diagnostic; no patch, source ancestry, collision clearance or native admission is established.'],'inputs_sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [base,fine,w,Path(__file__)]}}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
