from __future__ import annotations
import hashlib, itertools, json, sys, time
from pathlib import Path
import numpy as np

BASE=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207')
PT=BASE/'physical-transfer-209'
FIRST=PT/'bank32-f32-conditioning-001/output'
BANK=PT/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/bank-envelopes/bank-023-032-cube2um.npz'
OPERAND=PT/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/batch-007/bank-operands/bank-23-32.npz'
REPLAY=PT/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz'
REPLAY_REPORT=PT/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/report.json'
OUT=PT/'bank32-f32-conditioning-001/run-002-ulp-and-exact-gates'
assert not (OUT/'candidate-f32.npz').exists()
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
EXPECTED={'bank':'4cb25b1387af6caabb3479ff990ac27b49d1592817af508e8e3d78d9d6f29067',
          'operand':'509d823330bf404ebaf01ae1a95fd5966097f906f4600f4a8cf4d82161041179',
          'first_f32':'aca987996494fe155ca41bedb5f7658324fc211613bfd6807d6296a14f41da01',
          'first_f64':'113a3db12871eef6bda0d8cff4f119eb06114fd83a8a89ffdf8ef4fe40bd8506',
          'replay':'e9f0fffefa68f274e8d1c0a565d3d6cab1cfba31d13e085b8fdc79ca362185e5'}
for name,p in [('bank',BANK),('operand',OPERAND),('first_f32',FIRST/'candidate-f32.npz'),('first_f64',FIRST/'candidate-f64.npz'),('replay',REPLAY)]:
    if sha(p)!=EXPECTED[name]: raise RuntimeError(f'{name} source hash mismatch: {sha(p)}')

t0=time.monotonic()
a=np.load(FIRST/'candidate-f64.npz',allow_pickle=False); b=np.load(FIRST/'candidate-f32.npz',allow_pickle=False)
P64=np.asarray(a['points_world_m_f64'],np.float64); F64=np.asarray(a['faces_f64'],np.int64)
Y=np.asarray(b['points_world_m'],np.float32).copy(); F=np.asarray(b['faces'],np.int64)
if Y.shape!=(len(P64),3) or F.shape!=F64.shape or not np.isfinite(Y).all(): raise ValueError('pass-1 geometry shape/data mismatch')
emission=json.loads(str(b['float32_emission_json'][0])); m=np.asarray(emission['input_vertex_old_to_emitted'],np.int64)
ref=np.full(Y.shape,np.nan,np.float64)
for oldrow,outrow in enumerate(m):
    if outrow>=0: ref[int(outrow)]=P64[oldrow]
if not np.isfinite(ref).all(): raise ValueError('emitter map does not biject onto output rows')
Nref=np.cross(ref[F][:,1]-ref[F][:,0],ref[F][:,2]-ref[F][:,0])
if np.any(np.linalg.norm(Nref,axis=1)==0): raise ValueError('pass-1 F64 reference has zero-area faces')

def normals(q):
    t=q[F]
    return np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
def bad_rows(q):
    n=normals(q)
    d=np.einsum('ij,ij->i',Nref,n)
    return np.flatnonzero((np.linalg.norm(n,axis=1)==0)|(d<=0)),n,d

incident=[[] for _ in range(len(Y))]; neighbors=[set() for _ in range(len(Y))]
for fi,tr in enumerate(F):
    for v in tr:
        incident[int(v)].append(fi)
        neighbors[int(v)].update(int(w) for w in tr if int(w)!=int(v))
initial_bad,_,initial_dot=bad_rows(Y)
initial_witness=[]
for fi in initial_bad:
    initial_witness.append({'output_face_row':int(fi),'input_envelope_face_rows':np.asarray(b['candidate_face_ancestry_rows'][int(b['candidate_face_ancestry_offsets'][fi]):int(b['candidate_face_ancestry_offsets'][fi+1])],np.int64).tolist(),
      'vertices':F[fi].tolist(),'reference_normal':Nref[fi].tolist(),'initial_float32_normal':normals(Y)[fi].tolist(),'dot':float(initial_dot[fi])})
(OUT/'initial-orientation-witnesses.json').write_text(json.dumps({'schema':'numi.bank32-f32-ulp-repair-input-witness.v1','candidate_sha256':sha(FIRST/'candidate-f32.npz'),'count':len(initial_witness),'witnesses':initial_witness},indent=2)+'\n')

moves=[]
def ulp_candidates(q, max_radius=4):
    for radius in range(1,max_radius+1):
        for steps in itertools.product(range(-radius,radius+1),repeat=3):
            if max(abs(int(x)) for x in steps)!=radius: continue
            if steps==(0,0,0): continue
            v=q.copy()
            for ax,step in enumerate(steps):
                direction=np.float32(-np.inf if step<0 else np.inf)
                for _ in range(abs(int(step))): v[ax]=np.nextafter(v[ax],direction,dtype=np.float32)
            yield radius,steps,v

def local_eval(verts, proposals, qbase):
    # Check all faces incident to moved vertices, against the pinned pass-1 F64 geometry.
    rows=np.unique(np.concatenate([np.asarray(incident[int(v)],np.int64) for v in verts]))
    tri=Y[F[rows]].copy()
    for moved,q in proposals.items():
        mask=(F[rows]==int(moved))
        tri[mask]=np.asarray(q,dtype=np.float64)
    # Reject any face with coordinate-identical corners, then test orientation and area.
    repeat=np.all(tri[:,0]==tri[:,1],axis=1)|np.all(tri[:,1]==tri[:,2],axis=1)|np.all(tri[:,2]==tri[:,0],axis=1)
    nn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    dd=np.einsum('ij,ij->i',Nref[rows],nn)
    return rows, (not np.any(repeat) and np.all(np.linalg.norm(nn,axis=1)>0) and np.all(dd>0)), dd

while True:
    bad,curN,curDot=bad_rows(Y)
    if not len(bad): break
    badset=set(map(int,bad)); best=None
    target_vertices=sorted(set(int(v) for fi in bad for v in F[int(fi)]))
    # First try the smallest one-vertex ULP changes that make its whole incident star valid.
    for v in target_vertices:
        if not incident[v]: continue
        for radius,steps,q in ulp_candidates(Y[v],4):
            if float(np.linalg.norm(q.astype(np.float64)-ref[v]))>1e-7+1e-15: continue
            if any(np.array_equal(q,Y[n]) for n in neighbors[v]): continue
            rows,ok,dd=local_eval([v],{v:q},Y)
            if not ok: continue
            fixed=len(badset.intersection(map(int,rows)))
            dist=float(np.linalg.norm(q.astype(np.float64)-Y[v].astype(np.float64)))
            score=(-fixed,radius,sum(abs(int(x)) for x in steps),dist,int(v),tuple(map(int,steps)))
            if best is None or score<best[0]: best=(score,{v:q.copy()},rows,dd)
    # If no one-vertex move works, try paired ULP moves on one bad face's two corners.
    if best is None:
        for fi in bad:
            tri_ids=list(map(int,F[int(fi)]))
            for ix in range(3):
                for jx in range(ix+1,3):
                    v,w=tri_ids[ix],tri_ids[jx]
                    opts_v=list(ulp_candidates(Y[v],1));opts_w=list(ulp_candidates(Y[w],1))
                    for rv,sv,qv in opts_v:
                        if float(np.linalg.norm(qv.astype(np.float64)-ref[v]))>1e-7+1e-15 or any(np.array_equal(qv,Y[n]) for n in neighbors[v] if n!=w): continue
                        for rw,sw,qw in opts_w:
                            if float(np.linalg.norm(qw.astype(np.float64)-ref[w]))>1e-7+1e-15 or any(np.array_equal(qw,Y[n]) for n in neighbors[w] if n!=v): continue
                            if np.array_equal(qv,qw): continue
                            rows,ok,dd=local_eval([v,w],{v:qv,w:qw},Y)
                            if not ok: continue
                            fixed=len(badset.intersection(map(int,rows)))
                            dist=float(np.linalg.norm(qv.astype(np.float64)-Y[v].astype(np.float64))+np.linalg.norm(qw.astype(np.float64)-Y[w].astype(np.float64)))
                            score=(-fixed,max(rv,rw),sum(abs(int(x)) for x in sv)+sum(abs(int(x)) for x in sw),dist,v,w,tuple(map(int,sv)),tuple(map(int,sw)))
                            if best is None or score<best[0]: best=(score,{v:qv.copy(),w:qw.copy()},rows,dd)
    if best is None:
        break
    score,proposals,rows,dd=best
    for v,q in proposals.items():
        old=Y[v].copy(); Y[v]=q
        moves.append({'output_vertex_row':int(v),'pass1_reference_world_m':ref[v].tolist(),'before_f32':old.tolist(),'after_f32':q.tolist(),'before_bits':old.view(np.uint32).tolist(),'after_bits':q.view(np.uint32).tolist(),'delta_m':float(np.linalg.norm(q.astype(np.float64)-ref[v])),'faces_incident_checked':len(incident[v])})
    if len(moves)>64: raise RuntimeError('bounded ULP repair exceeded 64 vertex changes')

bad,finalN,finalDot=bad_rows(Y)
# Structural closure/nondegeneracy and complete input face ancestry are checked before exact geometry.
tris=Y[F].astype(np.float64); area=np.linalg.norm(np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]),axis=1)
E=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]); _,ec=np.unique(np.sort(E,axis=1),axis=0,return_counts=True)
volume=float(np.sum(np.einsum('ij,ij->i',tris[:,0],np.cross(tris[:,1],tris[:,2])))/6.0)
parent_off=np.asarray(a['output_vertex_input_parent_offsets'],np.int64);parent_rows=np.asarray(a['output_vertex_input_parent_rows'],np.int64)
orig=np.load(PT/'bank32-f32-conditioning-001/input/D-conditioner-resume-f64.npz',allow_pickle=False)['points_world_m']
max_input_parent_shift=0.0
for outrow in range(len(Y)):
    pr=parent_rows[parent_off[outrow]:parent_off[outrow+1]]
    if not len(pr): raise ValueError(f'output vertex {outrow} has no input ancestry')
    max_input_parent_shift=max(max_input_parent_shift,float(np.max(np.linalg.norm(orig[pr]-Y[outrow].astype(np.float64),axis=1))))
face_off=np.asarray(b['candidate_face_ancestry_offsets'],np.int64); face_rows=np.asarray(b['candidate_face_ancestry_rows'],np.int64)
all_face_parents=np.unique(face_rows)
structural={'zero_area_faces':int(np.sum(area==0)),'nonfinite_faces':int(np.sum(~np.isfinite(area))),'orientation_bad_faces':int(len(bad)),'closed_edge_incidence_2':bool(np.all(ec==2)),'edge_count':int(len(ec)),'signed_volume_m3':volume,'euler_characteristic':int(len(np.unique(F))-len(ec)+len(F)),'face_components_after_pass1':2,'input_face_ancestry_complete':bool(np.array_equal(all_face_parents,np.arange(32012,dtype=np.int64))),'max_vertex_delta_from_pass1_f64_reference_m':float(np.max(np.linalg.norm(Y.astype(np.float64)-ref,axis=1),initial=0.0)),'max_mapped_input_vertex_shift_m':max_input_parent_shift,'max_allowed_mapped_input_shift_m':1e-7,'ulp_moves':moves}
(OUT/'structural-result.json').write_text(json.dumps({'schema':'numi.bank32-ulp-final-structural.v1','candidate_parent_sha256':EXPECTED['first_f32'],'structural':structural},indent=2)+'\n')
if structural['zero_area_faces'] or structural['nonfinite_faces'] or structural['orientation_bad_faces'] or not structural['closed_edge_incidence_2'] or volume<=0 or not structural['input_face_ancestry_complete'] or max_input_parent_shift>1e-7:
    (OUT/'result.json').write_text(json.dumps({'status':'STRUCTURAL_FAIL','exact_self':'NOT_RUN','native_bone_cross':'NOT_RUN','native_bone_containment':'NOT_RUN','structural':structural,'elapsed_s':time.monotonic()-t0},indent=2)+'\n')
    print(json.dumps({'status':'STRUCTURAL_FAIL','initial_orientation_witnesses':len(initial_bad),'remaining_orientation_witnesses':len(bad),'moves':len(moves),'structural':{k:v for k,v in structural.items() if k!='ulp_moves'}},indent=2)); raise SystemExit(2)

# Preserve all pass-1 lineage and emit a fresh source-prepared F32 surface.
outd={k:np.asarray(b[k]) for k in b.files}
outd['points_world_m']=Y.astype(np.float32); outd['faces']=F.copy()
outd['bank32_final_ulp_vertex_operations_json']=np.asarray([json.dumps(moves,separators=(',',':'))])
outd['bank32_final_parent_candidate_sha256']=np.asarray([EXPECTED['first_f32']])
CAND=OUT/'candidate-f32.npz'; np.savez_compressed(CAND,**outd)
wide={k:np.asarray(a[k]) for k in a.files}
wide['points_world_m_f64']=Y.astype(np.float64); wide['faces_f64']=F.copy()
wide['bank32_final_ulp_vertex_operations_json']=np.asarray([json.dumps(moves,separators=(',',':'))])
wide['bank32_final_parent_candidate_sha256']=np.asarray([EXPECTED['first_f32']])
np.savez_compressed(OUT/'candidate-f64.npz',**wide)

# Use the existing CGAL exact owner: whole candidate self, exact cross-surface check, then source-minus-envelope containment.
sys.path.insert(0,str(BASE/'authoring-deps'))
import igl.copyleft.cgal as cgal
V=Y.astype(np.float64);Fout=F
sv, sf, self_pairs, sj, sim=cgal.remesh_self_intersections(V,Fout,detect_only=True,first_only=False)
self_pairs=np.asarray(self_pairs,np.int64).reshape(-1,2)
np.savez_compressed(OUT/'candidate-self-intersections.npz',face_pairs=self_pairs)
self_status='PASS' if len(self_pairs)==0 else 'FAIL'

rz=np.load(REPLAY,allow_pickle=False); rr=json.loads(REPLAY_REPORT.read_text())
idx=64
report_bank=next((x for x in rr['banks'] if int(x.get('index',-1))==48),None)
if report_bank is None or int(report_bank['body'])!=23 or int(report_bank['bone'])!=32 or int(report_bank['source_id'])!=80000048: raise ValueError('replay report logical index48 no longer identifies body23/bone32 source80000048')
if int(rz['bank_body'][idx])!=23 or int(rz['bank_bone'][idx])!=32 or int(rz['bank_source_id'][idx])!=80000048: raise ValueError('replay array row64 no longer identifies logical bank48 body23/bone32 source80000048')
vo=np.asarray(rz['bank_vertex_offsets'],np.int64); fo=np.asarray(rz['bank_face_offsets'],np.int64)
v0,v1=int(vo[idx]),int(vo[idx+1]); f0,f1=int(fo[idx]),int(fo[idx+1])
NB=rz['points_world_f32'][v0:v1].astype(np.float64); NBF=np.asarray(rz['faces'][f0:f1],np.int64)-v0
if (len(NB),len(NBF))!=(7257,14510): raise ValueError(f'unexpected replay-index48 geometry {len(NB)}/{len(NBF)}')
combinedV=np.vstack([NB,V]); combinedF=np.vstack([NBF,Fout+len(NB)])
cv,cf,cross_pairs,cj,cim=cgal.remesh_self_intersections(combinedV,combinedF,detect_only=True,first_only=False)
cross_pairs=np.asarray(cross_pairs,np.int64).reshape(-1,2)
native_face_count=len(NBF)
is_cross=((cross_pairs[:,0]<native_face_count)&(cross_pairs[:,1]>=native_face_count))|((cross_pairs[:,1]<native_face_count)&(cross_pairs[:,0]>=native_face_count))
actual_cross=cross_pairs[is_cross]
np.savez_compressed(OUT/'native-bone-envelope-intersections.npz',face_pairs=actual_cross,native_face_count=np.asarray([native_face_count]),candidate_face_count=np.asarray([len(Fout)]))

Dv,Df,Dj=cgal.mesh_boolean(NB,NBF,V,Fout,'difference')
Dv=np.asarray(Dv,np.float64); Df=np.asarray(Df,np.int64); Dj=np.asarray(Dj,np.int64)
if len(Df): np.savez_compressed(OUT/'native-bone-minus-conditioned-envelope.npz',points_world_m_f64=Dv,faces=Df,birth_face_rows=Dj)

exact={'self':{'status':self_status,'pair_count':int(len(self_pairs)),'witness_path':str(OUT/'candidate-self-intersections.npz')},'native_bone_cross':{'status':'PASS' if len(actual_cross)==0 else 'FAIL','pair_count':int(len(actual_cross)),'witness_path':str(OUT/'native-bone-envelope-intersections.npz'),'native_replay_report_index':48,'native_replay_array_row':64,'native_body':23,'native_bone':32,'native_vertex_count':len(NB),'native_face_count':len(NBF)},'native_bone_containment':{'status':'PASS' if len(Df)==0 else 'FAIL','difference_faces':int(len(Df)),'difference_vertices':int(len(Dv)),'witness_path':str(OUT/'native-bone-minus-conditioned-envelope.npz') if len(Df) else None,'operation':'CGAL exact-predicate Boolean native replay bank minus emitted conditioned envelope'},'native_replay_sha256':EXPECTED['replay'],'cgal_self_module_sha256':sha(Path(cgal.__file__))}
status='PASS_STRUCTURAL_SELF_AND_NATIVE_BONE_CONTAINMENT' if all(exact[k]['status']=='PASS' for k in ('self','native_bone_cross','native_bone_containment')) else 'FAIL_EXACT_GEOMETRY_GATE'
result={'schema':'numi.bank32-conditioned-envelope-qualification.v1','status':status,'input_hashes':{'bank_envelope':EXPECTED['bank'],'native_replay':EXPECTED['replay'],'parent_bank_operand':EXPECTED['operand'],'first_pass_candidate':EXPECTED['first_f32']},'candidate_f32_path':str(CAND),'candidate_f32_sha256':sha(CAND),'candidate_f64_path':str(OUT/'candidate-f64.npz'),'candidate_f64_sha256':sha(OUT/'candidate-f64.npz'),'structural':structural,'exact_gates':exact,'coverage_basis':'candidate is derived from the pinned 4 um L-infinity Minkowski envelope; actual replay-index48 bone minus candidate was explicitly tested','runtime':0,'accepted_physical_steps':0,'physical_admission':False,'elapsed_s':time.monotonic()-t0}
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':status,'candidate_sha256':result['candidate_f32_sha256'],'structural':{k:v for k,v in structural.items() if k!='ulp_moves'},'exact_gates':exact,'elapsed_s':result['elapsed_s']},indent=2),flush=True)
if status!='PASS_STRUCTURAL_SELF_AND_NATIVE_BONE_CONTAINMENT': raise SystemExit(2)
