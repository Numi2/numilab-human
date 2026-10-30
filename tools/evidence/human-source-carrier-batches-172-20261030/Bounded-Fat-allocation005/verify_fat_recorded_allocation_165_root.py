"""Independent invariant checks of recorded CSV flows; no allocation solver."""
from pathlib import Path
import csv,hashlib,json,time,argparse
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--result-directory',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--result-receipt',default='fat-source-only-carrier-transfer-005.json');args=ap.parse_args()
D=args.result_directory.resolve();P=Path(__file__).resolve().parent.parent.parent;H=P/'fine-physical-volume-706'
B=D.parent/'fat-853406-run-003-root';Q=B/'source-quadrature-162-root'
S=P/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv'
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def load(name,cols):
 a=np.loadtxt(D/name,delimiter=',',skiprows=1,ndmin=2)
 if a.ndim!=2 or a.shape[1]!=cols or not np.isfinite(a).all():raise ValueError('CSV shape/finite '+name)
 return a
def close(a,b,atol,rtol,what):
 err=np.abs(a-b)
 if np.any(err>np.maximum(atol,rtol*np.abs(b))):raise ValueError(what+' maximumerror '+str(err.max()))
 return float(err.max(initial=0))
start=time.monotonic()
receipt=D/args.result_receipt
r=json.loads(receipt.read_text())
assert r['status']=='PASS_SOURCE_ONLY_FAT_SOURCE_CARRIER_SHARED_CAPACITY_ALLOCATION'
assert r['recorded_flow_deterministic_replay']=='PASS' and r['physical_admission'] is False and r['native_steps']==0
for name,pin in r['flow_csv'].items():
 p=D/name
 assert sha(p)==pin['sha256'] and p.stat().st_size==pin['size_bytes']
source=read(S.with_name(S.name+'.tets.csv'));labels=read(S.with_name(S.name+'.labels.csv'));nodes=read(S.with_name(S.name+'.nodes.csv'))
globalrows=np.asarray([i for i,x in enumerate(source) if int(x['material_index'])==16])
pt=np.asarray([[int(x['parent_node'+str(k)]) for k in range(4)] for x in source])
px=np.asarray([[float(x['rest_'+axis+'_m']) for axis in 'xyz'] for x in nodes])
pm=np.asarray([float(source[g]['parent_mass_kg']) for g in globalrows])
pv=np.asarray([float(source[g]['parent_reference_volume_m3']) for g in globalrows])
regions=np.asarray([int(labels[g]['anatomical_region_id']) for g in globalrows])
parent=load('transport-source-parent.csv',11);child=load('transport-child-mass.csv',4)
assert len(parent)==14316 and len(child)==853406
assert np.array_equal(parent[:,0],np.arange(len(parent))) and np.array_equal(parent[:,1],globalrows) and np.all(parent[:,2]==16)
assert np.array_equal(parent[:,3],np.asarray([int(labels[g]['geometric_donor_body']) for g in globalrows]))
assert np.array_equal(parent[:,4],regions) and np.array_equal(parent[:,5],np.asarray([int(labels[g]['stable_source_face_id']) for g in globalrows]))
assert np.array_equal(parent[:,6],pm) and np.array_equal(parent[:,7],pv)
close(parent[:,8:],px[pt[globalrows]].mean(axis=1),1e-12,0,'source centroid')
assert np.array_equal(child[:,0],np.arange(len(child))) and np.all(child[:,1:]>0)
density=child[:,2]/child[:,1]
assert np.all(density>=850-1e-7) and np.all(density<=950+1e-7)
close(child[:,3],density,1e-8,1e-12,'child density')
raw=np.loadtxt(H/'combined-child-handoff-142/fat.node',skiprows=1,usecols=(0,1,2,3))
ids=raw[:,0].astype(np.int64);cx=raw[:,1:].astype(np.float32).astype(np.float64)
elements=np.loadtxt(H/'combined-child-handoff-142/fat.ele',skiprows=1)
ct=np.searchsorted(ids,elements[:,1:5].astype(np.int64));assert np.array_equal(ids[ct],elements[:,1:5].astype(np.int64)) and np.all(elements[:,5]==16)
cc=cx[ct].mean(axis=1)
dv=cx[ct][:,1:]-cx[ct][:,:1]
volume=np.abs(np.einsum('ij,ij->i',dv[:,0],np.cross(dv[:,1],dv[:,2]))/6)
close(volume,child[:,1],1e-15,1e-10,'recorded/reference F32 geometric volume')
assert r['input_child_node_sha256']==sha(H/'combined-child-handoff-142/fat.node') and r['input_child_element_sha256']==sha(H/'combined-child-handoff-142/fat.ele')
edge=load('transport-flow.csv',5);carrier=load('source-carrier-flow.csv',7)
ec=edge[:,0].astype(np.int64);ep=edge[:,1].astype(np.int64);fc=carrier[:,0].astype(np.int64);fp=carrier[:,1].astype(np.int64)
assert np.all(edge[:,2:4]>0) and np.all(edge[:,4]>=0) and np.all(carrier[:,4:6]>0) and np.all(carrier[:,6]>=0)
assert np.all(fc>=0)&np.all(fc<len(child))&np.all(fp>=0)&np.all(fp<len(parent))
assert np.all(ec>=0)&np.all(ec<len(child))&np.all(ep>=0)&np.all(ep<len(parent))
child_error=close(np.bincount(ec,weights=edge[:,2],minlength=len(child)),child[:,2],0,1e-10,'child inflow')
parent_error=close(np.bincount(ep,weights=edge[:,2],minlength=len(parent)),pm,1e-13,1e-10,'parent mass closure')
measure_error=close(np.bincount(ep,weights=edge[:,3],minlength=len(parent)),pv,1e-15,1e-10,'parent source measure closure')
close(edge[:,3],edge[:,2]*(pv/pm)[ep],1e-15,1e-10,'edge source measure')
with np.load(Q/'source-quadrature.npz',allow_pickle=False) as z:q={k:z[k] for k in z.files}
counts=np.maximum(np.diff(q['parent_offsets'].astype(np.int64)),1)
expanded_source=np.repeat(np.arange(len(parent)),counts)
expanded_carrier=np.full(len(expanded_source),np.iinfo(np.uint32).max,dtype=np.uint32)
sites=np.empty((len(expanded_source),3));expanded_weight=np.ones(len(expanded_source));n=0
for p,count in enumerate(counts):
 a,b=int(q['parent_offsets'][p]),int(q['parent_offsets'][p+1])
 if a==b:sites[n]=parent[p,8:]
 else:
  sites[n:n+count]=q['carrier_barycentric'][a:b]@px[pt[globalrows[p]]]
  expanded_carrier[n:n+count]=np.arange(a,b,dtype=np.uint32)
  expanded_weight[n:n+count]=q['carrier_weights'][a:b]
 n+=count
ef=carrier[:,2].astype(np.int64)
assert np.all(ef>=0) and np.all(ef<len(sites))
expanded_mass=pm[expanded_source]*expanded_weight
expanded_measure=pv[expanded_source]*expanded_weight
carrier_mass_error=close(np.bincount(ef,weights=carrier[:,4],minlength=len(sites)),expanded_mass,0,1e-10,'individual source carrier mass closure')
carrier_measure_error=close(np.bincount(ef,weights=carrier[:,5],minlength=len(sites)),expanded_measure,0,1e-10,'individual source carrier measure closure')
assert np.array_equal(expanded_source[ef],fp) and np.array_equal(expanded_carrier[ef],carrier[:,3].astype(np.uint32))
actual_distance=np.linalg.norm(sites[ef]-cc[fc],axis=1)
close(carrier[:,6],actual_distance,1e-12,0,'carrier actual distance')
assert np.all(actual_distance<=.019353+1e-12)
close(carrier[:,5],carrier[:,4]*(pv/pm)[fp],1e-15,1e-10,'carrier source measure')
key=(fc.astype(np.uint64)<<32)|fp.astype(np.uint64);order=np.argsort(key,kind='stable');unique,starts=np.unique(key[order],return_index=True)
ekey=(ec.astype(np.uint64)<<32)|ep.astype(np.uint64);eo=np.argsort(ekey,kind='stable')
assert np.array_equal(unique,ekey[eo]) and len(unique)==len(edge)
mass=np.add.reduceat(carrier[order,4],starts);measure=np.add.reduceat(carrier[order,5],starts)
close(mass,edge[eo,2],1e-13,1e-10,'aggregate carrier mass')
close(measure,edge[eo,3],1e-15,1e-10,'aggregate carrier measure')
close(np.add.reduceat(carrier[order,4]*carrier[order,6],starts)/mass,edge[eo,4],1e-12,1e-10,'aggregate weighted distance')
with np.load(H/'connected-region-mass-support-143-root/fat-connected-region-support.npz',allow_pickle=False) as z:ao=z['cell_admissible_region_offsets'];ai=z['cell_admissible_region_ids']
for reg in np.unique(regions):
 cells=np.searchsorted(ao,np.flatnonzero(ai==reg),side='right')-1
 allowed=np.zeros(len(child),dtype=bool);allowed[cells]=True
 assert np.all(allowed[ec[regions[ep]==reg]])
# Rebuild parent source-donor shares independently from actual nodal source mass.
source_rows=read(S.with_name(S.name+'.source.csv'));by_node={}
for row in source_rows:
 if int(row['material_index'])!=16:continue
 node=int(row['object_local_node']);donor=int(row['donor_body']);mass=float(row['mass_kg'])
 by_node.setdefault(node,[]).append((donor,mass))
splits={}
for p,g in enumerate(globalrows):
 for node in pt[g]:
  owners=by_node[int(node)];total=sum(x[1] for x in owners)
  for donor,mass in owners:splits[p,donor]=splits.get((p,donor),0)+pm[p]*.25*mass/total
pd=load('transport-source-parent-donor.csv',3);cd=load('transport-child-source-donor.csv',3)
assert len(splits)==len(pd) and np.all(pd[:,2]>0) and np.all(cd[:,2]>0)
assert set((int(p),int(d)) for p,d,_ in pd)==set(splits)
cd_keys=(cd[:,0].astype(np.uint64)<<32)|cd[:,1].astype(np.uint64)
assert len(np.unique(cd_keys))==len(cd) and set(map(int,np.unique(cd[:,1]))).issubset(set(map(int,np.unique(pd[:,1]))))
assert np.all(cd[:,0]>=0) and np.all(cd[:,0]<len(child))
close(np.bincount(cd[:,0].astype(np.int64),weights=cd[:,2],minlength=len(child)),child[:,2],1e-13,1e-10,'child donor total')
for p,d,m in pd:close(np.array([m]),np.array([splits[int(p),int(d)]]),1e-13,1e-10,'source donor split')
donor_error=0.
for donor in np.unique(pd[:,1].astype(np.int64)):
 fraction=np.zeros(len(parent));mask=pd[:,1]==donor
 fraction[pd[mask,0].astype(np.int64)]=pd[mask,2]/pm[pd[mask,0].astype(np.int64)]
 expected=np.bincount(ec,weights=edge[:,2]*fraction[ep],minlength=len(child))
 mask=cd[:,1]==donor;actual=np.zeros(len(child));actual[cd[mask,0].astype(np.int64)]=cd[mask,2]
 donor_error=max(donor_error,close(actual,expected,1e-13,1e-10,'child source donor composition'))
report={'status':'PASS_INDEPENDENT_FAT_RECORDED_MASS_AND_CARRIER_FLOW','physical_admission':False,'native_steps':0,'counts':{'source_parents':len(parent),'children':len(child),'aggregate_flow_edges':len(edge),'carrier_flow_edges':len(carrier)},'source_mass_kg':float(pm.sum()),'child_mass_kg':float(child[:,2].sum()),'density_min_kg_m3':float(density.min()),'density_max_kg_m3':float(density.max()),'maximum_carrier_distance_m':float(actual_distance.max()),'maximum_parent_mass_error_kg':parent_error,'maximum_source_carrier_mass_error_kg':carrier_mass_error,'maximum_source_carrier_measure_error_m3':carrier_measure_error,'maximum_child_mass_error_kg':child_error,'maximum_parent_measure_error_m3':measure_error,'maximum_source_donor_child_composition_error_kg':donor_error,'allocation_result_sha256':sha(receipt),'source_quadrature_sha256':sha(Q/'source-quadrature.bin'),'verifier_sha256':sha(Path(__file__)),'elapsed_s':time.monotonic()-start}
(D/'independent-recorded-allocation-verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2))
