#!/usr/bin/env python3
"""Classify current typed-volume exterior facets against O/I/D source facets."""
import hashlib, json
from pathlib import Path
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
T=H/'current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz'
I=H/'all-quality-1009-point-update-1018-root/actual-001/candidate-I-point-updated.npz'
O=H/'all-quality-937-b5-backtracking-root/proposal/candidate-O-typed-current.npz'
D=H/'all-quality-958-D014-root/proposal-003/candidate-D-typed-current.npz'
OUT=H/'joint-local-i-patch-preflight-002/boundary-face-classification-002'
PINS={'typed':'5d31c141b66628801a711c2e745f744f745611882a269e173eddfcd0e99d3372','I':'5557cbb88b46ea83d7977d60e1d362a10c48c290b77afb39b273efa29d32e323','O':'10169962e298586d63794bf94515ecc3a9b1ce91e66cc60566a6919d7c856ac3','D':'63cec05102fd2dc670d9220ec6472cd7d9ffa7085b4694719c605947086c7da9'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def pin(p): return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def canon(f):
 a=np.asarray(f,np.int64).reshape(-1,3).copy(); a.sort(axis=1); return a
def main():
 if OUT.exists(): raise FileExistsError(OUT)
 for name,p in [('typed',T),('I',I),('O',O),('D',D)]:
  if sha(p)!=PINS[name]: raise ValueError('input pin mismatch '+name)
 with np.load(T,allow_pickle=False) as z:
  t=z['volume_tetrahedra'].astype(np.int64); mat=z['material_index'].astype(np.int64); don=z['geometric_donor_body'].astype(np.int64); reg=z['anatomical_region_id'].astype(np.int64)
  imap=z['current_typed_role_source_row_to_joint_node_I'].astype(np.int64); omap=z['current_typed_role_source_row_to_joint_node_O'].astype(np.int64); dmap=z['current_typed_role_source_row_to_joint_node_D'].astype(np.int64)
 with np.load(I,allow_pickle=False) as z: fi=imap[z['faces'].astype(np.int64)]
 with np.load(O,allow_pickle=False) as z: fo=omap[z['faces'].astype(np.int64)]
 with np.load(D,allow_pickle=False) as z: fd=dmap[z['faces'].astype(np.int64)]
 raw=np.concatenate((t[:,[1,2,3]],t[:,[0,3,2]],t[:,[0,1,3]],t[:,[0,2,1]]),axis=0)
 uniq,first,count=np.unique(canon(raw),axis=0,return_index=True,return_counts=True)
 if np.any((count!=1)&(count!=2)): raise ValueError('unexpected current tet face valence')
 bfaces=uniq[count==1]; bowner=(first[count==1]%len(t)).astype(np.int64)
 src=canon(np.concatenate((fo,fi,fd),axis=0)); src=np.unique(src,axis=0)
 joined=np.concatenate((bfaces,src),axis=0); both,inv=np.unique(joined,axis=0,return_inverse=True)
 nb=len(bfaces); bseen=np.zeros(len(both),bool); sseen=np.zeros(len(both),bool)
 bseen[inv[:nb]]=True; sseen[inv[nb:]]=True
 extra_mask=~sseen[inv[:nb]]; missing_mask=~bseen[inv[nb:]]
 extra=bfaces[extra_mask]; extra_owner=bowner[extra_mask]; missing=src[missing_mask]
 isurf_nodes=np.unique(np.r_[fi.reshape(-1),fo.reshape(-1),fd.reshape(-1)])
 boundary_nodes=np.unique(bfaces.reshape(-1))
 node_membership=np.isin(extra,isurf_nodes)
 ucat=np.concatenate((uniq,canon(fi)),axis=0); uall,uinv=np.unique(ucat,axis=0,return_inverse=True)
 tet_cofaces=np.zeros(len(uall),np.int8); tet_cofaces[uinv[:len(uniq)]]=count.astype(np.int8)
 icount=tet_cofaces[uinv[len(uniq):]]
 labels,lc=np.unique(np.stack((mat[extra_owner],don[extra_owner],reg[extra_owner]),axis=1),axis=0,return_counts=True) if len(extra_owner) else (np.empty((0,3),np.int64),np.empty(0,np.int64))
 OUT.mkdir(parents=True)
 np.savez_compressed(OUT/'boundary-difference-witnesses.npz',extra_tet_boundary_faces=extra,extra_tet_boundary_owner_rows=extra_owner,missing_from_tet_boundary_O_I_D_faces=missing,I_face_canonical=canon(fi),I_face_coface_count=icount,extra_boundary_face_source_node_membership=node_membership.astype(np.uint8))
 report={'schema':'numi.h706.current1020-tet-vs-O-I-D-boundary-face-classification.v2','status':'PASS_EXACT_BOUNDARY_MISMATCH_CLASSIFIED','physical_admission':False,'native_steps':0,'inputs':{k:pin(p) for k,p in {'typed433':T,'I':I,'O':O,'D':D}.items()},'tet_mesh':{'nodes':311768,'tets':len(t),'unique_tet_faces':len(uniq),'one_coface_boundary_faces':len(bfaces),'two_coface_faces':int(np.count_nonzero(count==2)),'face_incidence_max':int(count.max(initial=0))},'source_surface_faces':{'I':len(fi),'O':len(fo),'D':len(fd),'OID_unique_canonical_faces':len(src)},'exact_comparison':{'common_OID_tet_boundary_faces':int(np.count_nonzero(bseen&sseen)),'tet_boundary_extra_vs_OID':len(extra),'OID_faces_missing_from_tet_boundary':len(missing),'tet_boundary_nodes_not_on_any_OID_source_surface':len(set(map(int,boundary_nodes))-set(map(int,isurf_nodes))),'I_face_coface_counts':{str(int(v)):int(np.count_nonzero(icount==v)) for v in np.unique(icount)},'extra_boundary_face_source_node_membership_counts':{'all3':int(np.count_nonzero(np.all(node_membership,axis=1))),'some':int(np.count_nonzero(np.any(node_membership,axis=1)&~np.all(node_membership,axis=1))),'none':int(np.count_nonzero(~np.any(node_membership,axis=1))) }},'extra_boundary_labels':{'columns':['material_index','geometric_donor_body','anatomical_region_id'],'rows':[{'labels':row.tolist(),'face_count':int(n)} for row,n in zip(labels,lc)],'first_extra_faces':extra[:12].tolist(),'first_extra_owner_tets':extra_owner[:12].tolist()},'missing_OID_faces_first12':missing[:12].tolist(),'witnesses':pin(OUT/'boundary-difference-witnesses.npz'),'interpretation':'Exact face identity only. The actual one-coface tet boundary is external and must remain fixed. Differences from O/I/D face tessellations are not interpreted as defects without source ancestry; I-face coface counts are reported separately.'}
 rp=OUT/'report.json'; rp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
