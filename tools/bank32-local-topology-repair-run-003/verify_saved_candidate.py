from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
PT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
BASE=PT.parent
RUN=PT/'bank32-f32-conditioning-001/run-003-local-topology-repair'
OUT=RUN/'readback'
CAND=RUN/'result/candidate-f32.npz'
F64=RUN/'result/candidate-f64.npz'
VIEW=RUN/'result/BONE-derived-126000033-source.npz'
PARENT=PT/'bank32-f32-conditioning-001/run-002-ulp-and-exact-gates/candidate-f32.npz'
RESULT=RUN/'result/result.json'
REPLAY=PT/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz'
REPORT=PT/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/report.json'
EXPECTED={'candidate':'22bd7be7e85fe63ebd6717ff1abdda9c847964cdc470644fba92d643e693fd39','parent':'da83ed13621e25624625cabaf15452748c31b4a129a265874c28f13afc437b92','replay':'e9f0fffefa68f274e8d1c0a565d3d6cab1cfba31d13e085b8fdc79ca362185e5','cgal':'677e9b1ec585722d78685ad5e00e091ceb5ecef2ba2eecb877c5c8eb153fa1b5'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,p in [('candidate',CAND),('parent',PARENT),('replay',REPLAY)]:
    if sha(p)!=EXPECTED[name]: raise RuntimeError(f'{name} SHA mismatch')
d=np.load(CAND,allow_pickle=False); v=np.asarray(d['points_world_m'],np.float32).astype(np.float64); f=np.asarray(d['faces'],np.int64)
if v.shape!=(15821,3) or f.shape!=(31638,3): raise RuntimeError('candidate dimensions changed')
if not np.array_equal(v.astype(np.float32),d['points_world_m']): raise RuntimeError('candidate points are not exact Float32 values')
# Check closed topology, positive oriented volume, no zero-area/duplicate faces.
tri=v[f]; normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); areas=np.linalg.norm(normals,axis=1)
if not np.isfinite(areas).all() or np.any(areas==0): raise RuntimeError('zero/nonfinite emitted triangle')
edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]); ue,counts=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)
if not np.all(counts==2): raise RuntimeError('closed edge incidence failed')
if len(np.unique(np.sort(f,axis=1),axis=0))!=len(f): raise RuntimeError('duplicate face')
used=np.unique(f); euler=int(len(used)-len(ue)+len(f)); volume=float(np.sum(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])))/6)
if euler!=2 or volume<=0: raise RuntimeError('topology or orientation changed')
# Verify every source face row, union tuple, and node row still has a descendant.
fo=np.asarray(d['candidate_face_ancestry_offsets'],np.int64); fr=np.asarray(d['candidate_face_ancestry_rows'],np.int64)
face_universe=np.unique(fr)
if not np.array_equal(face_universe,np.arange(32012,dtype=np.int64)): raise RuntimeError('input face ancestry is incomplete')
uo=np.asarray(d['union_face_ancestry_offsets'],np.int64); ur=np.asarray(d['union_face_ancestry_run_original_ids'],np.int64); uf=np.asarray(d['union_face_ancestry_face_ids'],np.int64); uw=np.asarray(d['union_face_ancestry_rows'],np.int64)
union_universe=set(zip(map(int,ur),map(int,uf),map(int,uw)))
if len(union_universe)!=32012 or any(rid!=100000064 for rid,_,_ in union_universe): raise RuntimeError('union/source face ancestry incomplete or wrong owner')
no=np.asarray(d['output_vertex_input_parent_offsets'],np.int64); nr=np.asarray(d['output_vertex_input_parent_rows'],np.int64)
node_universe=np.unique(nr)
if not np.array_equal(node_universe,np.arange(16008,dtype=np.int64)): raise RuntimeError('source node ancestry incomplete')
# Derived-view exact row/face/coordinate/ancestry parity.
s=np.load(VIEW,allow_pickle=False); mapfaces=np.asarray(s['source_candidate_face_rows'],np.int64); mapverts=np.asarray(s['source_candidate_vertex_rows'],np.int64)
if len(mapfaces)!=4 or len(mapverts)!=len(s['points_world_m']): raise RuntimeError('unexpected derived source-view dimensions')
if not np.array_equal(np.asarray(s['points_world_m'],np.float32),v[mapverts].astype(np.float32)): raise RuntimeError('derived view points mismatch candidate')
if not np.array_equal(mapverts[np.asarray(s['faces'],np.int64)],f[mapfaces]): raise RuntimeError('derived view faces mismatch candidate')
if not np.all(d['face_source_original_id'][mapfaces]==126000033): raise RuntimeError('derived source identity missing in candidate rows')
sfo=np.asarray(s['face_ancestry_offsets'],np.int64); sfr=np.asarray(s['face_ancestry_input_cache_rows'],np.int64)
for localr,globalr in enumerate(mapfaces):
    a=set(map(int,sfr[sfo[localr]:sfo[localr+1]])); b=set(map(int,fr[fo[globalr]:fo[globalr+1]]))
    if a!=b: raise RuntimeError('derived view face ancestry differs from candidate')
# Exact CGAL owner and actual native replay bank row64.
sys.path.insert(0,str(BASE/'authoring-deps'))
import igl.copyleft.cgal as cgal
if sha(Path(cgal.__file__))!=EXPECTED['cgal']: raise RuntimeError('CGAL exact owner hash changed')
ans=cgal.remesh_self_intersections(v,f,detect_only=True,first_only=False); pairs=np.asarray(ans[2],np.int64).reshape(-1,2)
np.savez_compressed(OUT/'saved-candidate-self-pairs.npz',face_pairs=pairs)
rr=json.loads(REPORT.read_text()); row=next(x for x in rr['banks'] if int(x.get('index',-1))==48); rz=np.load(REPLAY,allow_pickle=False); idx=64
if (int(row['body']),int(row['bone']),int(row['source_id']))!=(23,32,80000048) or (int(rz['bank_body'][idx]),int(rz['bank_bone'][idx]),int(rz['bank_source_id'][idx]))!=(23,32,80000048): raise RuntimeError('native bank identity row mapping mismatch')
vo=np.asarray(rz['bank_vertex_offsets'],np.int64); xo=np.asarray(rz['bank_face_offsets'],np.int64); v0,v1=int(vo[idx]),int(vo[idx+1]); f0,f1=int(xo[idx]),int(xo[idx+1])
bv=np.asarray(rz['points_world_f32'][v0:v1],np.float32).astype(np.float64); bf=np.asarray(rz['faces'][f0:f1],np.int64)-v0
combined_v=np.vstack([bv,v]); combined_f=np.vstack([bf,f+len(bv)])
cross=cgal.remesh_self_intersections(combined_v,combined_f,detect_only=True,first_only=False); cp=np.asarray(cross[2],np.int64).reshape(-1,2); is_cross=((cp[:,0]<len(bf))&(cp[:,1]>=len(bf)))|((cp[:,1]<len(bf))&(cp[:,0]>=len(bf))); cp=cp[is_cross]
np.savez_compressed(OUT/'saved-candidate-native-cross-pairs.npz',face_pairs=cp)
dv,df,dj=cgal.mesh_boolean(bv,bf,v,f,'difference'); df=np.asarray(df,np.int64).reshape(-1,3); dv=np.asarray(dv,np.float64).reshape(-1,3)
if len(df): np.savez_compressed(OUT/'saved-candidate-native-difference.npz',points_world_m_f64=dv,faces=df)
report={'schema':'numi.bank32-local-repair-saved-file-readback.v1','status':'PASS' if len(pairs)==0 and len(cp)==0 and len(df)==0 else 'FAIL','candidate_path':str(CAND),'candidate_sha256':sha(CAND),'derived_source_view_sha256':sha(VIEW),'parent_candidate_sha256':EXPECTED['parent'],'native_replay_sha256':EXPECTED['replay'],'cgal_module_sha256':EXPECTED['cgal'],'structural':{'vertices':len(v),'faces':len(f),'closed_edge_incidence_2':bool(np.all(counts==2)),'euler_characteristic':euler,'signed_volume_m3':volume,'zero_area_faces':int(np.sum(areas==0))},'lineage':{'source_input_face_rows':len(face_universe),'union_source_face_tuples':len(union_universe),'source_node_rows':len(node_universe),'derived_view_face_rows':len(mapfaces),'derived_view_exact_geometry_and_ancestry_match':True},'exact_gates':{'self_pairs':int(len(pairs)),'native_bone_cross_pairs':int(len(cp)),'native_bone_minus_envelope_difference_faces':int(len(df)),'native_replay_report_index':48,'native_replay_array_row':64,'body':23,'bone':32},'physical_admission':False,'runtime':0,'accepted_physical_steps':0}
(OUT/'result.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps(report,sort_keys=True,indent=2),flush=True)
if report['status']!='PASS': raise SystemExit(2)
