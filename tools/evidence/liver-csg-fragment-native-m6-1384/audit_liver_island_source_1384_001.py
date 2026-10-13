#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,struct,sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
R=A/'liver-retirement-native-023-004-removed-alias/resting-anatomy-receipt.json'
O=A/'liver-island-source-audit-1384-001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not O.exists();O.mkdir()
 r=json.loads(R.read_text());p=Path(r['payload']['path']);blob=p.read_bytes()
 assert sha(p)==r['payload']['sha256']
 h=struct.unpack_from('<8s5I32s',blob,0);rows=[struct.unpack_from('<8I',blob,60+32*i) for i in range(h[2])]
 v0=60+32*h[2];i0=v0+24*h[3];ps=[];fs=[];owners=[];locals=[];base=0
 for s in range(14,22):
  row=next(x for x in rows if x[5]==s);assert row[0]==20 and row[6]==1
  points=np.frombuffer(blob,dtype='<f4',offset=v0+24*row[1],count=6*row[2]).reshape((-1,6))[:,:3].copy()
  faces=(np.frombuffer(blob,dtype='<u4',offset=i0+4*row[3],count=row[4]).reshape((-1,3))-row[1]).astype(np.int64)
  ps.append(points);fs.append(faces+base);owners.extend([s]*len(faces));locals.extend(range(len(faces)));base+=len(points)
 p32=np.concatenate(ps);f=np.concatenate(fs);u,inv=np.unique(p32,axis=0,return_inverse=True);wf=inv[f]
 edges=np.concatenate([wf[:,[0,1]],wf[:,[1,2]],wf[:,[2,0]]]);e=np.unique(np.sort(edges,axis=1),axis=0)
 graph=coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(u),len(u))).tocsr()
 n,labels=connected_components(graph,directed=False);counts=[]
 owners=np.asarray(owners);locals=np.asarray(locals)
 for label in range(n):
  ids=np.flatnonzero(labels[wf[:,0]]==label)
  if not len(ids):continue
  vert=np.unique(wf[ids]);points=u[vert].astype(float);tri=u[wf[ids]].astype(float);center=points.mean(axis=0)
  vv=tri-center;vol=float(np.sum(np.einsum('ij,ij->i',vv[:,0],np.cross(vv[:,1],vv[:,2])))/6)
  counts.append({'label':int(label),'faces':len(ids),'vertices':len(vert),'signed_volume_ml':vol*1e6,
   'bbox_extent_mm':(np.ptp(points,axis=0)*1000).tolist(),'centroid_source_body20_m':center.tolist(),
   'owner_face_counts':{str(int(s)):int(np.sum(owners[ids]==s)) for s in np.unique(owners[ids])},
   'global_face_indices':ids.tolist(),'row_local_faces':{str(int(s)):locals[ids][owners[ids]==s].tolist() for s in np.unique(owners[ids])}})
 counts.sort(key=lambda c:c['faces'],reverse=True)
 assert len(counts)==2 and counts[1]['faces']==48 and counts[1]['owner_face_counts']=={'19':48}
 im=np.unique(wf[counts[1]['global_face_indices']]);mm=np.unique(wf[counts[0]['global_face_indices']])
 distance,near=cKDTree(u[mm].astype(float)).query(u[im].astype(float))
 report={'source_receipt':str(R),'source_receipt_sha256':sha(R),'source_payload':str(p),'source_payload_sha256':sha(p),
  'helper_sha256':sha(Path(__file__)),'connected_components':counts,'island_to_main_vertex_distance_mm':{
   'min':float(distance.min()*1000),'max':float(distance.max()*1000),'median':float(np.median(distance)*1000)},
  'gpu_or_source_mutation':False,'classification':'Pending source-construction lineage; separate closed island cannot be assumed an intended liver compartment.'}
 (O/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 np.savez_compressed(O/'geometry.npz',points=u,faces=wf,owners=owners,local_faces=locals)
 print(json.dumps({k:v for k,v in report.items() if k!='connected_components'}))
 for c in counts:print(json.dumps({k:v for k,v in c.items() if k not in ('global_face_indices','row_local_faces')}))
if __name__=='__main__':main()
