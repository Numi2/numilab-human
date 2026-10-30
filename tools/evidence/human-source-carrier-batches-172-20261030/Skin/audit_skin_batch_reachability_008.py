import csv, hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
D=Path(__file__).resolve().parent
P=D.parent.parent; H=P/'fine-physical-volume-706'; S=P/'native-physical-parent-mass-330/native-source-identity-control-003'
T=S/'compiled-parent-cell-mass.csv'
OUT=D/'child-reachability-008'; OUT.mkdir(exist_ok=False)
Q=D/'source-quadrature-007/source-quadrature.npz'; R=.019353

def sha(p):
    with open(p,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def csvrows(p):
    with open(p,newline='') as f: return list(csv.DictReader(f))
start=time.monotonic()
labels=csvrows(T.with_name(T.name+'.labels.csv')); tets=csvrows(T.with_name(T.name+'.tets.csv')); nodes=csvrows(T.with_name(T.name+'.nodes.csv'))
global_skin=np.flatnonzero(np.fromiter((int(x['material_index'])==15 for x in labels),dtype=bool,count=len(labels))).astype(np.int64)
role_labels=[labels[int(g)] for g in global_skin]; role_tets=[tets[int(g)] for g in global_skin]
px=np.asarray([[float(r['rest_'+a+'_m']) for a in 'xyz'] for r in nodes],dtype=np.float64)
pt=np.asarray([[int(r['parent_node'+str(k)]) for k in range(4)] for r in role_tets],dtype=np.int64)
regions=np.asarray([int(r['anatomical_region_id']) for r in role_labels],dtype=np.int32)
with np.load(Q,allow_pickle=False) as z: q={k:z[k] for k in z.files}
assert np.array_equal(q['source_parent_tet_rows'],np.arange(len(global_skin),dtype=np.uint32))
assert np.array_equal(q['global_source330_tet_rows'],global_skin.astype(np.uint32))
off=q['parent_offsets'].astype(np.int64); w=q['carrier_weights']; b=q['carrier_barycentric']
target=q['carrier_target_child_rows'].astype(np.int64); stored_d=q['carrier_target_distance_m']
assert off.shape==(len(global_skin)+1,) and off[0]==0 and off[-1]==len(w)==len(b)==len(target)==len(stored_d)
assert np.all(np.diff(off)>=0) and np.all(w>0) and np.all(b>0) and np.all(np.abs(b.sum(axis=1)-1)<2e-12)
raw=np.loadtxt(H/'skin010-source004-admission-073-retry001/skin.node',skiprows=1,usecols=(0,1,2,3))
child_ids=raw[:,0].astype(np.int64); cx=raw[:,1:4].astype(np.float32).astype(np.float64)
ele=np.loadtxt(H/'skin010-source004-admission-073-retry001/skin.ele',skiprows=1)
assert ele.shape==(743497,6)
conn=np.searchsorted(child_ids,ele[:,1:5].astype(np.int64)); assert np.array_equal(child_ids[conn],ele[:,1:5].astype(np.int64))
cc=cx[conn].mean(axis=1)
with np.load(H/'connected-region-mass-support-143-root/skin-connected-region-support.npz',allow_pickle=False) as z:
    ao=z['cell_admissible_region_offsets'].astype(np.int64); ai=z['cell_admissible_region_ids'].astype(np.int32)
assert len(ao)==len(cc)+1 and ao[0]==0 and ao[-1]==len(ai)
child_material=ele[:,5].astype(np.int32); assert np.all(child_material==15)
# Unmodified source parents retain their original centroid site. Modified parents use
# their positive source quadrature; its 5% reserve preserves the old site representation.
site_xyz=[]; site_parent=[]; site_region=[]; source_centroid=np.empty((len(pt),3))
for p,tet in enumerate(pt):
    tetxyz=px[tet]; source_centroid[p]=tetxyz.mean(axis=0)
    a,z=off[p:p+2]
    if a==z:
        site_xyz.append(source_centroid[p]); site_parent.append(p); site_region.append(regions[p])
    else:
        xyz=b[a:z]@tetxyz
        if abs(float(w[a:z].sum())-1)>2e-12: raise ValueError(f'source parent {p}: weights do not close')
        barymoment=w[a:z]@b[a:z]; worldmoment=w[a:z]@xyz
        if np.max(np.abs(barymoment-.25))>2e-10 or np.linalg.norm(worldmoment-source_centroid[p])>1e-11:
            raise ValueError(f'source parent {p}: first moment fails')
        site_xyz.extend(xyz); site_parent.extend([p]*(z-a)); site_region.extend([regions[p]]*(z-a))
site_xyz=np.asarray(site_xyz); site_parent=np.asarray(site_parent,dtype=np.int32); site_region=np.asarray(site_region,dtype=np.int32)
carrier_parent=np.repeat(np.arange(len(global_skin),dtype=np.int32),np.diff(off))
carrier_xyz=np.empty_like(b)
for p in np.flatnonzero(np.diff(off)):
    a,z=off[p:p+2]; carrier_xyz[a:z]=b[a:z]@px[pt[p]]
actual_witness_distance=np.linalg.norm(carrier_xyz-cc[target],axis=1)
assert np.all(actual_witness_distance<R) and np.max(np.abs(actual_witness_distance-stored_d))<1e-12
for p in np.flatnonzero(np.diff(off)):
    a,z=off[p:p+2]; reg=int(regions[p])
    for child in target[a:z]:
        if reg not in ai[ao[child]:ao[child+1]]: raise ValueError(f'carrier target {child} is not region-admissible for parent {p}')
# Index current child cells by every authenticated admissible region.
slot_child=np.repeat(np.arange(len(cc),dtype=np.int32),np.diff(ao))
sort=np.argsort(ai,kind='stable'); reg_sorted=ai[sort]; child_sorted=slot_child[sort]
unique,starts,counts=np.unique(reg_sorted,return_index=True,return_counts=True)
children_by_region={int(r):np.unique(child_sorted[s:s+n]) for r,s,n in zip(unique,starts,counts)}
# Source-side complete recensus: every source site versus the current child cells that
# admit its parent region, under the unchanged transport radius.
source_nearest=np.full(len(site_xyz),np.inf); source_witness=np.full(len(site_xyz),-1,dtype=np.int32)
for reg,children in children_by_region.items():
    ix=np.flatnonzero(site_region==reg)
    if len(ix)==0 or len(children)==0: continue
    dd,jj=cKDTree(cc[children]).query(site_xyz[ix],k=1)
    source_nearest[ix]=dd; source_witness[ix]=children[jj]
source_miss=np.flatnonzero(source_nearest>R)
# Receiver-side full recensus across all 743,497 cells and each cell's admitted regions.
nearest=np.full(len(cc),np.inf); nearest_site=np.full(len(cc),-1,dtype=np.int32)
for reg,s,n in zip(unique,starts,counts):
    children=np.unique(child_sorted[s:s+n]); ix=np.flatnonzero(site_region==int(reg))
    if not len(children) or not len(ix): continue
    dd,jj=cKDTree(site_xyz[ix]).query(cc[children],k=1)
    improve=dd<nearest[children]; row=children[improve]
    nearest[row]=dd[improve]; nearest_site[row]=ix[jj[improve]]
holes=np.flatnonzero(nearest>R)
# If any holes remain, preserve their full authenticated P1 source pool as a witness.
parent_pool=[];pool_off=[0];pool_child=[]
if len(holes):
    with np.load(H/'connected-region-mass-support-143-root/skin-connected-region-support.npz',allow_pickle=False) as z:
        go=z['cell_group_offsets']; gi=z['cell_group_ids']; po=z['cell_group_parent_tet_offsets']; pr=z['cell_group_parent_tet_rows']
    for c in holes:
        rows=set()
        for slot in gi[go[c]:go[c+1]]: rows.update(int(global_skin[int(r)]) for r in pr[po[slot]:po[slot+1]])
        parent_pool.extend(sorted(rows)); pool_child.extend([int(c)]*len(rows)); pool_off.append(len(parent_pool))
report={'schema':'numi.human.skin-source-quadrature-complete-reachability.v1','status':'SOURCE_ONLY_COMPLETE_SKIN_SOURCE_AND_RECEIVER_REACHABILITY','physical_admission':False,'native_steps':0,'role':'Skin','source_parent_count':len(global_skin),'child_cell_count':len(cc),'modified_source_parent_count':int(np.count_nonzero(np.diff(off))),'carrier_count':len(w),'source_site_count':len(site_xyz),'max_distance_m':R,'source_side':{'uncovered_site_count':int(len(source_miss)),'maximum_nearest_admissible_child_distance_m':float(source_nearest.max()),'worst_sites':[{'site_row':int(i),'role_parent_row':int(site_parent[i]),'global_source330_tet_row':int(global_skin[site_parent[i]]),'distance_m':float(source_nearest[i]),'nearest_child_row':int(source_witness[i])} for i in source_miss[:100]]},'receiver_side':{'uncovered_child_count':int(len(holes)),'maximum_nearest_admissible_source_site_distance_m':float(nearest.max()),'worst_children':[{'child_row':int(c),'distance_m':float(nearest[c]),'nearest_source_role_parent_row':int(site_parent[nearest_site[c]]) if nearest_site[c]>=0 else -1,'nearest_source_global_tet_row':int(global_skin[site_parent[nearest_site[c]]]) if nearest_site[c]>=0 else -1,'admissible_regions':ai[ao[c]:ao[c+1]].astype(int).tolist()} for c in holes[:100]]},'witness_children_addressed_by_quadrature':int(len(np.unique(target))),'uncovered_child_p1_pool_entries':len(parent_pool),'inputs':{'quadrature_npz_sha256':sha(Q),'quadrature_binary_sha256':sha(D/'source-quadrature-007/source-quadrature.bin'),'quadrature_receipt_sha256':sha(D/'source-quadrature-007/receipt.json'),'source_mass_csv_sha256':sha(T),'source_nodes_sha256':sha(T.with_name(T.name+'.nodes.csv')),'source_tets_sha256':sha(T.with_name(T.name+'.tets.csv')),'source_labels_sha256':sha(T.with_name(T.name+'.labels.csv')),'child_node_sha256':sha(H/'skin010-source004-admission-073-retry001/skin.node'),'child_ele_sha256':sha(H/'skin010-source004-admission-073-retry001/skin.ele'),'support143_sha256':sha(H/'connected-region-mass-support-143-root/skin-connected-region-support.npz')},'elapsed_s':time.monotonic()-start}
np.savez_compressed(OUT/'uncovered-child-witnesses.npz',child_rows=holes.astype(np.uint32),nearest_source_role_parent_rows=np.asarray([site_parent[i] for i in nearest_site[holes]],dtype=np.int32),nearest_source_global_tet_rows=np.asarray([global_skin[site_parent[i]] for i in nearest_site[holes]],dtype=np.int64),nearest_distance_m=nearest[holes],parent_pool_offsets=np.asarray(pool_off,dtype=np.int64),parent_pool_global_tet_rows=np.asarray(parent_pool,dtype=np.int64),parent_pool_child_rows=np.asarray(pool_child,dtype=np.int32))
report['outputs']={str(OUT/'uncovered-child-witnesses.npz'):{'sha256':sha(OUT/'uncovered-child-witnesses.npz'),'size_bytes':(OUT/'uncovered-child-witnesses.npz').stat().st_size}}
(OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':report['status'],'source_side':report['source_side'],'receiver_side':report['receiver_side'],'elapsed_s':report['elapsed_s'],'report_sha256':sha(OUT/'report.json')},indent=2))
