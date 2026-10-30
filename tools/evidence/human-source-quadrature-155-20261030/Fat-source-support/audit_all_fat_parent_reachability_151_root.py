import csv,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
D=Path(__file__).resolve().parent
P=D.parent.parent;H=P/'fine-physical-volume-706'
S=P/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv'
t=time.monotonic()
def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
nodes=read(S.with_name(S.name+'.nodes.csv'));tets=read(S.with_name(S.name+'.tets.csv'));labels=read(S.with_name(S.name+'.labels.csv'))
xyz=np.asarray([[float(r['rest_'+a+'_m']) for a in 'xyz'] for r in nodes])
fullrows=np.asarray([i for i,r in enumerate(tets) if int(r['material_index'])==16])
pt=np.asarray([[int(tets[i]['parent_node'+str(k)]) for k in range(4)] for i in fullrows])
pc=xyz[pt].mean(axis=1)
preg=np.asarray([int(labels[i]['anatomical_region_id']) for i in fullrows])
pn=np.loadtxt(H/'combined-child-handoff-142/fat.node',skiprows=1,usecols=(0,1,2,3))
ndids=pn[:,0].astype(np.int64);cp=pn[:,1:4].astype(np.float32).astype(np.float64)
el=np.loadtxt(H/'combined-child-handoff-142/fat.ele',skiprows=1)
ct=np.searchsorted(ndids,el[:,1:5].astype(np.int64))
assert np.array_equal(ndids[ct],el[:,1:5].astype(np.int64)) and np.all(el[:,5]==16)
cc=cp[ct].mean(axis=1)
with np.load(H/'connected-region-mass-support-143-root/fat-connected-region-support.npz',allow_pickle=False) as z:
 ao=z['cell_admissible_region_offsets'];ai=z['cell_admissible_region_ids'];origin=z['admissibility_source_support_sha256'].item()
cell_for_slot=np.repeat(np.arange(len(ct),dtype=np.int32),np.diff(ao))
order=np.argsort(ai,kind='stable')
regions,starts,counts=np.unique(ai[order],return_index=True,return_counts=True)
child_sorted=cell_for_slot[order];miss=[];maximum=0
for reg,st,ctcount in zip(regions,starts,counts):
 pp=np.flatnonzero(preg==reg)
 if not len(pp):continue
 ci=np.unique(child_sorted[st:st+ctcount])
 ds,ii=cKDTree(cc[ci]).query(pc[pp])
 maximum=max(maximum,float(ds.max()))
 for i,d,j in zip(pp,ds,ii):
  if d>.019353:
   child=int(ci[j]);r=labels[fullrows[i]]
   miss.append({'fat_parent_row':int(i),'parent_tet_row':int(fullrows[i]),'region':int(reg),'stable_source_face_id':int(r['stable_source_face_id']),'mass_kg':float(tets[fullrows[i]]['mass_kg']) if 'mass_kg' in tets[0] else None,'nearest_admissible_child_row':child,'nearest_distance_m':float(d),'parent_centroid_m':pc[i].tolist(),'child_centroid_m':cc[child].tolist(),'source_tet_nodes':pt[i].tolist(),'source_tet_vertices_m':xyz[pt[i]].tolist(),'source_diameter_m':float(np.linalg.norm(xyz[pt[i]][:,None,:]-xyz[pt[i]][None,:,:],axis=2).max())})
out={'status':'SOURCE_ONLY_ALL_FAT_PARENT_REACHABILITY_CENSUS','physical_admission':False,'native_steps':0,'parent_cells':len(pt),'child_cells':len(ct),'allowed_region_slots':len(ai),'source_regions':len(regions),'max_transport_distance_m':.019353,'parents_without_any_admissible_child_within_bound':len(miss),'max_nearest_distance_m':maximum,'witnesses':miss,'support_origin_sha256':origin,'elapsed_s':time.monotonic()-t}
(D/'all-parent-reachability-census.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2))
