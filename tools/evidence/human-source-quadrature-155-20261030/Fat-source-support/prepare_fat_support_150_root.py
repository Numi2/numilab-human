#!/usr/bin/env python3
"""Prepare the existing FSSUP090 role-local Fat support, not physical admission."""
import csv,hashlib,json,struct,time
from pathlib import Path
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
H=P/'fine-physical-volume-706'
D=Path(__file__).resolve().parent
M=H/'combined-child-handoff-142'
A=H/'fat-current-semantic-support-141-root'
AD=H/'connected-region-mass-support-143-root'
S=P/'native-physical-parent-mass-330/native-source-identity-control-003'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def require(b,msg):
 if not b:raise ValueError(msg)
start=time.monotonic()
expected={
 M/'fat.node':'ba5d39714bbef252d8c76c438083b432c2e1956a7a298b1b9b48e690aaa4d2e4',
 M/'fat.ele':'69cefbb10501154206d2065e80a5babe5cf6b561644f3a12f68c142e5f073ec6',
 A/'fat-current-semantic-support.npz':'f3081e59fd88a88b48aa3d9caa8153609f6c24fcc0aac32d069c634ddad12b2b',
 AD/'fat-connected-region-support.npz':'0e47882aab9219230bc263eb4b23759830666908fefdb90f66a970e9655e2d13',
 AD/'receipt.json':'055a1fb7f909205248b6d8ebe160ef8ea1c743f335615fc93dd14192728de246',
 AD/'independent-verification.json':'b0940d6954779767009f257da92de587a6cc57d58c304b925177a3a3e972ea2d',
 S/'compiled-parent-cell-mass.csv':'c7823c42d64a2154a34c87cc0d4d202f0d5256f79217e74f5f62bcf1d5b31679',
 S/'compiled-parent-cell-mass.csv.labels.csv':'99d643f71a735ee49223cdcd7b863d2d0978b8cc490a138bad4a63012438a457',
 S/'compiled-parent-cell-mass.csv.tets.csv':'82e20067454748a9eca91bc6f06ab343677c051a139797ae051c64022bd87992',
 S/'registered-skin-poses-100.json':'0c6e10d411604cb317eb5a9da234da568d683e6cc0276ac184f80fd29067b80a'}
for p,h in expected.items():require(sha(p)==h,'SHA mismatch '+str(p))
with np.load(A/'fat-current-semantic-support.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
with np.load(AD/'fat-connected-region-support.npz',allow_pickle=False) as z:b={k:z[k] for k in z.files}
require(all(k in b and np.array_equal(v,b[k]) for k,v in a.items()),'adjacency modified P1 atlas')
ar=json.loads((A/'receipt.json').read_text())
require(ar['status']=='PASS_SOURCE_ONLY_COMPLETE_SAME_MATERIAL_FAT_SEMANTIC_SUPPORT' and ar['physical_admission'] is False,'Fat support receipt')
material=16;n=853406
off=a['cell_group_offsets'].astype(np.int64);gi=a['cell_group_ids'].astype(np.int64)
score=a['cell_group_p1_score_numerators']/a['cell_group_p1_score_denominators']
tuples=[tuple(map(int,x)) for x in a['group_tuples']]
require(len(tuples)==314 and len(set(tuples))==314 and off.shape==(n+1,) and off[0]==0 and off[-1]==len(gi) and np.all(np.diff(off)>0),'P1 domain')
require(np.all(np.isfinite(score)) and np.all(score>0),'P1 fractions')
err=float(np.max(np.abs(np.add.reduceat(score,off[:-1])-1)))
require(err<1e-12,'P1 sums')
with open(S/'compiled-parent-cell-mass.csv.labels.csv',newline='') as f:
 rd=csv.DictReader(f);names=rd.fieldnames
 require(names==['parent_tet_row','material_index','geometric_donor_body','anatomical_region_id','stable_source_face_id'],'source schema')
 rows=[tuple(int(r[k]) for k in names) for r in rd]
require(len(rows)==28632 and all(r[0]==i for i,r in enumerate(rows)),'source rows')
parents=[r for r in rows if r[1]==material]
groups={}
for i,r in enumerate(parents):groups.setdefault(r[1:4],[]).append(i)
keys=sorted(groups);lookup={k:i for i,k in enumerate(keys)}
require(len(parents)==14316 and len(keys)==314 and set(tuples)==set(keys),'source material identities')
remap=np.asarray([lookup[t] for t in tuples],dtype=np.uint32)
sg=remap[gi]
# Both producers use sorted unique groups per child. Preserve that order.
same=np.ones(len(sg)-1,dtype=bool);same[off[1:-1]-1]=False
require(np.all(sg[1:][same]>sg[:-1][same]),'noncanonical child group ordering')
mx=np.maximum.reduceat(score,off[:-1])
cell=np.repeat(np.arange(n,dtype=np.int64),np.diff(off))
candidates=np.where(score==mx[cell],sg,np.iinfo(np.uint32).max)
primary=np.minimum.reduceat(candidates,off[:-1])
po=[0];pr=[];fo=[0];fr=[]
stable=np.asarray([r[4] for r in parents],dtype=np.uint32)
for k in keys:
 rr=groups[k];pr.extend(rr);po.append(len(pr));faces=sorted(set(map(int,stable[rr])));fr.extend(faces);fo.append(len(fr))
ado=b['cell_admissible_region_offsets'];adr=b['cell_admissible_region_ids']
require(ado.shape==(n+1,) and ado[0]==0 and ado[-1]==len(adr) and np.all(np.diff(ado)>0),'adjacency domain')
require(str(b['admissibility_source_support_sha256'].item())==expected[A/'fat-current-semantic-support.npz'] and int(b['admissibility_max_source_region_hops'].item())==1,'adjacency proof')
source_regions={k[2] for k in keys}
require(set(map(int,np.unique(adr))).issubset(source_regions),'unknown source region')
base=hashlib.sha256(('current-fat-semantic-support.v1\n'+expected[A/'fat-current-semantic-support.npz']+'\n'+sha(A/'receipt.json')+'\n'+expected[S/'compiled-parent-cell-mass.csv.labels.csv']+'\n'+expected[S/'compiled-parent-cell-mass.csv.tets.csv']).encode()).hexdigest()
digest=hashlib.sha256(('current-fat-connected-region-support-143.v1\n'+base+'\n'+expected[AD/'fat-connected-region-support.npz']+'\n'+expected[AD/'receipt.json']+'\n'+expected[AD/'independent-verification.json']).encode()).hexdigest()
def raw(x,t):return np.asarray(x,dtype=t).tobytes()
g=np.asarray(keys,dtype=np.uint32)
# Reserved internal derived identity namespace; no claim that this is one source face.
derived=np.asarray([(1<<63)|(material<<32)|i for i in range(len(keys))],dtype=np.uint64)
out=D/'semantic-support.bin'
require(not out.exists(),'output already exists')
with open(out,'wb') as f:
 f.write(b'FSSUP090');f.write(struct.pack('<6Q',n,len(sg),len(keys),len(parents),len(pr),len(fr)))
 arrays=(off,sg,score,g[:,0],g[:,1],g[:,2],derived,stable,po,pr,fo,fr,np.full(n,material,dtype=np.uint32),g[primary,1],g[primary,2])
 for i,v in enumerate(arrays):f.write(raw(v,'<f8' if i==2 else '<u8' if i==6 else '<u4'))
 f.write(digest.encode());f.write(b'FSSADJ01');f.write(struct.pack('<QI',len(adr),1))
 f.write(expected[A/'fat-current-semantic-support.npz'].encode());f.write(raw(ado,'<u4'));f.write(raw(adr,'<u4'))
# Independent byte readback using the established field sequence.
sizes=[(n+1,4),(len(sg),4),(len(sg),8),(len(keys),4),(len(keys),4),(len(keys),4),(len(keys),8),(len(parents),4),(len(keys)+1,4),(len(pr),4),(len(keys)+1,4),(len(fr),4),(n,4),(n,4),(n,4)]
require(out.stat().st_size==8+48+sum(c*s for c,s in sizes)+64+8+12+64+(n+1+len(adr))*4,'binary extent')
manifest={'status':'PASS_SOURCE_ONLY_FAT_CONNECTED_REGION_SUPPORT_PREPARATION','role':'Fat','physical_admission':False,'native_steps':0,
 'counts':{'child_cells':n,'p1_slots':len(sg),'atlas_groups':len(tuples),'source_fat_groups':len(keys),'source_fat_parent_tets':len(parents),'admissible_region_slots':len(adr)},
 'support_sha256':digest,'base_p1_support_sha256':base,'binary_path':str(out),'binary_sha256':sha(out),'binary_size':out.stat().st_size,
 'inputs':{str(p):{'sha256':h,'size':p.stat().st_size} for p,h in expected.items()},'producer_sha256':sha(Path(__file__)),
 'p1_score_sum_max_error':err,'density_bounds_kg_m3':{'material16':[850.,950.]},'transport_distance_m':.019353,
 'derived_group_identity_policy':'Reserved 64-bit namespace highbit|material<<32|source_group_index. This identifies a derived full source-face union, not a measured source face.',
 'semantic_policy':'Original source330 filtered Fat rows/keys and source-face unions; P1 scores are soft preferences. Authenticated one-hop same-material shared-face region permissions. No blood mass added.',
 'allocation_solver':{'name':'deterministic_dinic_feasibility','cost_optimality_certified':False},
 'scope':'Fat support preparation only, not allocation, paired closure, native admission or motion.',
 'elapsed_s':time.monotonic()-start}
(D/'preparation-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:manifest[k] for k in ('status','counts','binary_sha256','binary_size','elapsed_s')},indent=2))
