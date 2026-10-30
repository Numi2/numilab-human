"""Independent binary/geometry/ledger verification of the Skin source quadrature."""
from pathlib import Path
import csv,hashlib,json,struct,time
import numpy as np
D=Path(__file__).resolve().parent
Q=D/'source-quadrature-007';P=D.parent.parent;H=P/'fine-physical-volume-706'
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
t=time.monotonic();r=json.loads((Q/'receipt.json').read_text())
assert r['schema']=='numi.human.source-tetra-centroid-carrier-quadrature.v1' and r['physical_admission'] is False and r['native_steps']==0
assert all(sha(Path(p))==h for p,h in r['inputs'].items())
assert all(sha(Path(p))==v['sha256'] for p,v in r['outputs'].items())
blob=(Q/'source-quadrature.bin').read_bytes();assert blob[:9]==b'FSSCARR01'
n,k=struct.unpack_from('<QQ',blob,9);assert n==14316 and k==r['counts']['carriers']
offset=25
def take(dtype,shape):
 global offset
 count=int(np.prod(shape));a=np.frombuffer(blob,dtype,count,offset=offset).reshape(shape).copy();offset+=count*np.dtype(dtype).itemsize
 return a
binary={'source_parent_tet_rows':take('<u4',(n,)),'global_source330_tet_rows':take('<u4',(n,)),'parent_offsets':take('<u4',(n+1,)),'carrier_weights':take('<f8',(k,)),'carrier_barycentric':take('<f8',(k,4)),'carrier_target_child_rows':take('<u4',(k,)),'carrier_target_distance_m':take('<f8',(k,))}
assert offset==len(blob)
with np.load(Q/'source-quadrature.npz',allow_pickle=False) as z:
 assert set(z.files)==set(binary)
 assert all(np.array_equal(z[key],val) for key,val in binary.items())
source=P/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv'
labs=read(source.with_name(source.name+'.labels.csv'));tn=read(source.with_name(source.name+'.tets.csv'));nd=read(source.with_name(source.name+'.nodes.csv'))
full=np.asarray([i for i,x in enumerate(labs) if int(x['material_index'])==15])
assert np.array_equal(full,binary['global_source330_tet_rows']) and np.array_equal(np.arange(n),binary['source_parent_tet_rows'])
sx=np.asarray([[float(x['rest_'+a+'_m']) for a in 'xyz'] for x in nd])
source_tet=np.asarray([[int(x['parent_node'+str(j)]) for j in range(4)] for x in tn])
raw=np.loadtxt(H/'combined-child-handoff-142/skin.node',skiprows=1,usecols=(0,1,2,3));ids=raw[:,0].astype(np.int64);cx=raw[:,1:].astype(np.float32).astype(np.float64)
el=np.loadtxt(H/'combined-child-handoff-142/skin.ele',skiprows=1);et=np.searchsorted(ids,el[:,1:5].astype(np.int64))
assert np.array_equal(ids[et],el[:,1:5].astype(np.int64)) and np.all(el[:,5]==15)
cc=cx[et].mean(axis=1)
with np.load(H/'connected-region-mass-support-143-root/skin-connected-region-support.npz',allow_pickle=False) as z:
 ao=z['cell_admissible_region_offsets'];ai=z['cell_admissible_region_ids']
def validate_packet(packet):
 off=packet['parent_offsets'];ww=packet['carrier_weights'];bb=packet['carrier_barycentric'];targets=packet['carrier_target_child_rows'];stored=packet['carrier_target_distance_m']
 if off.shape!=(n+1,) or off[0]!=0 or off[-1]!=len(ww) or np.any(np.diff(off.astype(np.int64))<0):raise ValueError('CSR')
 if bb.shape!=(len(ww),4) or np.any(~np.isfinite(ww)) or np.any(~np.isfinite(bb)) or np.any(ww<=0) or np.any(bb<=0) or np.max(np.abs(bb.sum(axis=1)-1))>1e-12:raise ValueError('weights or barycentric')
 if np.any(targets>=len(et)):raise ValueError('target range')
 maximum_distance=0.;maximum_moment=0.;fallback=[]
 for i in np.flatnonzero(np.diff(off.astype(np.int64))):
  a,b=int(off[i]),int(off[i+1]);w=ww[a:b];bs=bb[a:b];c=targets[a:b];globalrow=full[i]
  error=max(abs(float(w.sum())-1),float(np.max(np.abs(w@bs-.25))))
  if abs(float(w.sum())-1)>2e-12 or np.max(np.abs(w@bs-.25))>2e-10:raise ValueError('source mass/volume/firstmoment closure')
  moment_world=w@(bs@sx[source_tet[globalrow]])
  if np.linalg.norm(moment_world-sx[source_tet[globalrow]].mean(axis=0))>1e-11:raise ValueError('source world firstmoment')
  reg=int(labs[globalrow]['anatomical_region_id'])
  if any(reg not in set(map(int,ai[ao[j]:ao[j+1]])) for j in c):raise ValueError('inadmissible source region')
  xyz=bs@sx[source_tet[globalrow]];distance=np.linalg.norm(xyz-cc[c],axis=1)
  if np.any(distance>.019353) or np.max(np.abs(stored[a:b]-distance))>1e-12:raise ValueError('source carrier target radius')
  maximum_distance=max(maximum_distance,float(distance.max()));maximum_moment=max(maximum_moment,error);fallback.append(int(i))
 if fallback!=sorted(x['parent_row'] for x in r['parents']):raise ValueError('fallback parent list')
 return maximum_distance,maximum_moment
distance,moment=validate_packet(binary)
def rejected(name,field,change):
 copy={k:v.copy() for k,v in binary.items()};change(copy[field])
 try:validate_packet(copy)
 except ValueError:return name
 raise AssertionError('accepted tampered '+name)
controls=[rejected('negative_weight','carrier_weights',lambda a:a.__setitem__(0,-a[0])),rejected('broken_moment','carrier_barycentric',lambda a:a.__setitem__((0,0),a[0,0]+.01)),rejected('out_of_range_child','carrier_target_child_rows',lambda a:a.__setitem__(0,len(et))),rejected('false_distance','carrier_target_distance_m',lambda a:a.__setitem__(0,a[0]+.001))]
report={'status':'PASS_INDEPENDENT_SKIN_BATCH_POSITIVE_SOURCE_QUADRATURE','physical_admission':False,'native_steps':0,'allocation':'NOT_RUN_WITH_QUADRATURE','counts':{'original_parent_rows':n,'carrierized_parent_rows':r['counts']['carrierized_parents'],'carriers':k},'maximum_target_distance_m':distance,'maximum_mass_fraction_and_barycentric_moment_error':moment,'rejected_controls':controls,'producer_receipt_sha256':sha(Q/'receipt.json'),'binary_sha256':sha(Q/'source-quadrature.bin'),'verifier_sha256':sha(Path(__file__)),'elapsed_s':time.monotonic()-t}
(Q/'independent-verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2))
