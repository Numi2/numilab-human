#!/usr/bin/env python3
from pathlib import Path
import sys, types, json, hashlib, time, runpy, importlib.util
from fractions import Fraction
import numpy as np

D=Path('/Users/numi/Work/numi-human-connected-surface-069/D')
W=D/'connected-obstacle-207/physical-transfer-209/geometry-worker/D-connectivity-218/limiting-O-edge-audit-001/quality-failure-inventory-001/contraction-batch-001'
FINAL=W/'shape-repair-batch-005/contraction-001/candidate.npz'
BASE=D/'connected-obstacle-207/envelope-033-exact-seams/candidate.npz'
DELTA=W/'shape-repair-batch-005/contraction-001/original-o033-delta-001/face-delta.npz'
OUT=W/'shape-repair-batch-005/contraction-001/final-audit-006'
TOOLS=D/'publication-206-stage-001/tools'
BANKHELPER=TOOLS/'human_skin_fat_bone_audit.py'
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
EXACTPATH=E/'incremental-exact-audit-owner-253/src/numilab_human/cardiac_cavity_intersections.py'
OUT.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def records(tris,face_ids):
    pts=tuple(tuple(Fraction.from_float(float(x)) for x in p) for p in np.asarray(tris).reshape(-1,3))
    faces=np.arange(len(pts),dtype=np.int64).reshape(-1,3)
    raw=exact._records(pts,faces)
    return [(r[0],r[1],r[2],int(face_ids[i]),r[4]) for i,r in enumerate(raw)]
def self_pairs_touched(flip,rows):
    rows=set(map(int,rows));cand=[]
    for i in sorted(rows):
        rec=flip.record(i,flip.F[i])
        if rec is None:raise ValueError(f'exact Float32 degenerate changed face {i}')
        lo=np.asarray(rec[1],dtype=float);hi=np.asarray(rec[2],dtype=float)
        near=np.flatnonzero(np.all(flip.hi>=lo,axis=1)&np.all(flip.lo<=hi,axis=1))
        for j in set(map(int,near))|rows:
            if j==i or (j in rows and j<i):continue
            other=flip.recs[j]
            if other is None:continue
            cand.append((rec,other) if i<j else (other,rec))
    return exact._audit_record_pairs(cand,same_surface=True)
z=np.load(FINAL,allow_pickle=False);data={k:z[k] for k in z.files}
bz=np.load(BASE,allow_pickle=False);base={k:bz[k] for k in bz.files}
P=np.asarray(data['points_world_m'],dtype=np.float32).astype(np.float64)
F=np.asarray(data['source_O_faces'],dtype=np.int64)
Falias=np.asarray(data['faces'],dtype=np.int64)
if not np.array_equal(F,Falias):raise ValueError('faces and source_O_faces do not agree')
if not np.array_equal(P.astype(np.float32).view(np.uint32),np.asarray(base['points_world_m'],dtype=np.float32).view(np.uint32)):raise ValueError('candidate point coordinates differ from unchanged 15e base')
Fbase=np.asarray(base['source_O_faces'],dtype=np.int64)
dz=np.load(DELTA,allow_pickle=False)
fmap=np.asarray(dz['final_face_to_O033_face_row'],dtype=np.int64)
changed=np.asarray(dz['changed_final_face_rows_vs_O033'],dtype=np.int64)
if fmap.shape!=(len(F),) or np.any(fmap>=len(Fbase)) or len(np.unique(fmap[fmap>=0]))!=np.count_nonzero(fmap>=0):raise ValueError('candidate-to-original-O033 face fingerprint map invalid')
if not np.array_equal(np.flatnonzero(fmap<0),changed):raise ValueError('changed rows do not equal unmapped final face identities')
def oriented_key(t):
    a,b,c=map(int,t);return min((a,b,c),(b,c,a),(c,a,b))
for i,j in enumerate(fmap):
    if j>=0 and oriented_key(F[i])!=oriented_key(Fbase[j]):raise ValueError('mapped unchanged oriented face differs from original O033')
unchanged=int(np.count_nonzero(fmap>=0))
if 'O033_to_candidate_vertex_row' not in data or len(data['O033_to_candidate_vertex_row'])!=len(P):raise ValueError('missing explicit O033 vertex endpoint map')
vmap=np.asarray(data['O033_to_candidate_vertex_row'],dtype=np.int64)
if np.any(vmap<0) or np.any(vmap>=len(P)) or not np.array_equal(vmap[vmap],vmap):raise ValueError('O033 endpoint map is not range-valid and transitively composed')
if 'face_ancestry_offsets' not in data or len(data['face_ancestry_offsets'])!=len(F)+1:raise ValueError('missing full per-face ancestry CSR')
if not np.all(np.diff(np.asarray(data['face_ancestry_offsets'],np.int64))>0):raise ValueError('some final face has no source ancestry')
# Exact self audit of all changed faces against every overlapping O face.
pkg=types.ModuleType('numi_tools');pkg.__path__=[str(EXACTPATH.parent)];sys.modules['numi_tools']=pkg
for name,attrs in [('cardiac_cavity_geometry',{'analyze_topology':lambda *a,**k:None}),('model',{'ImportError':RuntimeError}),('physiology',{'canonical':lambda x:x})]:
    m=types.ModuleType('numi_tools.'+name);m.__dict__.update(attrs);sys.modules[m.__name__]=m
from numi_tools import cardiac_cavity_intersections as exact
flipfile=D/'connected-obstacle-207/physical-transfer-209/root-edge-flips/local_float32_edge_flip.py'
spec=importlib.util.spec_from_file_location('edgeflip_for_final_o_gate',flipfile);fm=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fm;spec.loader.exec_module(fm)
flip=fm.Flips(P,F)
if any(r is None for r in flip.recs):raise ValueError('one or more final F32 faces are exactly degenerate')
self_result=self_pairs_touched(flip,changed)
start=time.monotonic()
# Pinned all-185 bank arrays and transformations.
helper=runpy.run_path(str(BANKHELPER))
T,ids,gids,bodies,matrices,offsets,positions,bankkeys,pins,_=helper['read_bank_inputs'](BANKDIR,E)
local_cache={};bank_reports=[]
surface_faces=F[changed]
for n,(body0,bone0) in enumerate(bankkeys,1):
    body,bone=int(body0),int(bone0)
    if body not in local_cache:local_cache[body]=(P-positions[body])@matrices[body].T-offsets[body]
    S=local_cache[body][surface_faces]
    mask=(ids[:,0]==body)&(ids[:,1]==bone)
    B=T[mask];stable_ids=ids[mask,3]
    if not len(B):raise ValueError(f'empty typed bank {body}/{bone}')
    slo,shi=S.min(1),S.max(1);blo,bhi=B.min((0,1)),B.max((0,1))
    active=np.flatnonzero(np.all(shi>=blo,axis=1)&np.all(slo<=bhi,axis=1))
    if not len(active):
        item={'body':body,'stable_bone_id':bone,'bank_triangles':len(B),'changed_surface_aabb_active':0,'bone_aabb_active':0,'exact_aabb_candidate_pairs':0,'crossing_pairs':[]}
    else:
        rlo,rhi=slo[active].min(0),shi[active].max(0)
        bactive=np.flatnonzero(np.all(B.max(1)>=rlo,axis=1)&np.all(B.min(1)<=rhi,axis=1))
        sr=records(S[active],changed[active]);br=records(B[bactive],stable_ids[bactive])
        res=exact._audit_pair(sr,br,same_surface=False)
        item={'body':body,'stable_bone_id':bone,'bank_triangles':len(B),'changed_surface_aabb_active':len(active),'bone_aabb_active':len(bactive),'exact_aabb_candidate_pairs':res['aabb_candidate_pairs'],'crossing_pairs':res['triangle_pairs']}
    bank_reports.append(item)
    (OUT/'progress.json').write_text(json.dumps({'completed_banks':n,'expected_banks':len(bankkeys),'elapsed_seconds':time.monotonic()-start,'last_body':body,'last_bone':bone})+'\n')
qtri=P[F];edges=np.stack([np.linalg.norm(qtri[:,1]-qtri[:,0],axis=1),np.linalg.norm(qtri[:,2]-qtri[:,1],axis=1),np.linalg.norm(qtri[:,0]-qtri[:,2],axis=1)],axis=1);cross=np.linalg.norm(np.cross(qtri[:,1]-qtri[:,0],qtri[:,2]-qtri[:,0]),axis=1);quality=cross/edges.max(1)**2;alt=cross/edges.max(1)
report={'schema':'numi.connectedfit-O-final-shape-repair-original-O033-exact-self-185bone-audit.v1',
 'status':'COMPLETE_CHANGED_FACE_SELF_AND_185_TYPED_BONE_AUDIT_VS_ORIGINAL_O033',
 'candidate_path':str(FINAL),'candidate_sha256':sha(FINAL),
 'baseline_candidate_path':str(BASE),'baseline_candidate_sha256':sha(BASE),'original_O033_delta_path':str(DELTA),'original_O033_delta_sha256':sha(DELTA),
 'exact_predicate_path':str(EXACTPATH),'exact_predicate_sha256':sha(EXACTPATH),
 'typed_bank_parser_path':str(BANKHELPER),'typed_bank_parser_sha256':sha(BANKHELPER),
 'bank_input_pins':pins,'expected_typed_banks':185,'typed_banks_checked':len(bank_reports),
 'final_face_count':len(F),'baseline_face_count':len(Fbase),'final_point_count':len(P),
 'changed_final_face_count':len(changed),'unchanged_baseline_face_count':unchanged,
 'changed_final_face_rows':changed.tolist(),'removed_baseline_face_count':len(Fbase)-unchanged,
 'face_map':'final_face_to_O033_face_row comes from exact oriented cyclic triangle identity against pinned original O033; reversed winding counts as changed; -1 rows are audited.',
 'vertex_map':'O033_to_candidate_vertex_row explicitly maps every original O033 vertex to its retained contraction endpoint; range and endpoint idempotence checked; full point array remains un-compacted.',
 'per_face_source_ancestry_csr_rows':int(len(data['face_ancestry_parent_face_rows'])),
 'faces_and_source_O_faces_equal':True,'positions_bitwise_equal_to_baseline':True,
 'exact_changed_face_self_pair_count':self_result['count'],'exact_changed_face_self_pairs':self_result['triangle_pairs'],'self_aabb_candidate_pairs':self_result['aabb_candidate_pairs'],
 'typed_bone_aabb_active_bank_count':sum(x['changed_surface_aabb_active']>0 for x in bank_reports),
 'exact_changed_face_bone_crossing_pair_count':sum(len(x['crossing_pairs']) for x in bank_reports),
 'banks':bank_reports,
 'quality_definition':'Float32 cross-product norm divided by squared longest edge; dimensionless.',
 'faces_below_1e-3':int(np.sum(quality<1e-3)),'minimum_quality':float(quality.min()),
 'faces_below_0p25um_altitude':int(np.sum(alt<0.25e-6)),'minimum_altitude_um':float(alt.min()*1e6),
 'limitations':['Checks changed final surface faces against all typed bone surfaces using exact rational triangle predicates; unchanged faces are byte-identical to the pinned O033 baseline.','This is a surface crossing and mesh conditioning gate, not closed-solid admission, layer thickness, positive volume-cell, mass, native dynamics, or physical qualification.'],
 'elapsed_seconds':time.monotonic()-start}
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(OUT/'progress.json').unlink(missing_ok=True)
print(json.dumps({k:v for k,v in report.items() if k!='banks'}))
