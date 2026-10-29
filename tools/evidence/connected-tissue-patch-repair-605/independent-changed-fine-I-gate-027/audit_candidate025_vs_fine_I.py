#!/usr/bin/env python3
from pathlib import Path
import collections,csv,gc,hashlib,importlib.util,json,sys,time
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
OUT=Path(__file__).resolve().parent
C=P/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/fine-I-bone-barrier-run-024/closed603-merge-025/candidate-f32-linked-remesh-typed.npz'
BASE=P/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/candidate-f32.npz'
I=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/dnew-vs-i296-inventory-001/i328-cavity-audit-001/fill-003/I-f32-conditioning-367/condition-288-pair-connected-remesh-001-run-002/result/candidate-f32-typed.npz'
I_RECEIPT=I.parent/'receipt.json'
I_REGISTERED=P/'full-current-surface-registration-399/sidecars/registered-I.csv'
R608=P/'changed-face-all185-bone-gate-608/report.json'
HELP=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
started=time.monotonic()
expected={'candidate':'9790ed1fa49bacff36ea169c7b8354344e40f1df33f604c006bcfb7991723a2a','baseline':'e8e332fb4ab421db6e44f0f5fe638e378646fd63438a318dcf94ad84b7a8ce96','fine_I':'5d46ee99df2613e9f30a43075af95a6402beffe26392cd9f017d8c1a154bf410','registered_I':'31b7be649979c468584dc13626b69887d7d0d1a7e804403649585b51765f49dd'}
# Pin the exact inputs; report hash is checked from live input and emitted below.
actual={'candidate':sha(C),'baseline':sha(BASE),'fine_I':sha(I),'registered_I':sha(I_REGISTERED),'exact_owner_script':sha(HELP)}
assert all(actual[k]==expected[k] for k in ('candidate','baseline','fine_I','registered_I')),(actual,expected)
with np.load(BASE,allow_pickle=False) as z: base_source=set(map(int,z['face_source_original_id']))
with np.load(C,allow_pickle=False) as z:
 dpoints=z['points_world_m'].copy(); dfaces=z['faces'].copy(); dsource=z['face_source_original_id'].copy()
 dfaceid=z['face_source_face_id'].copy(); dbody=z['face_source_body'].copy(); dlayer=z['face_source_layer'].copy()
 did0=z['face_source_identity0'].copy(); did1=z['face_source_identity1'].copy()
new_rows=np.flatnonzero(~np.isin(dsource,np.asarray(sorted(base_source),np.int64))).astype(np.int64)
with np.load(I,allow_pickle=False) as z:
 ipoints=z['points_world_m'].copy(); ifaces=z['faces'].copy()
 isource=z['face_source_original_id'].copy(); ifaceid=z['face_source_face_id'].copy()
 ibody=z['face_source_body'].copy(); ilayer=z['face_source_layer'].copy()
 iid0=z['face_source_identity0'].copy(); iid1=z['face_source_identity1'].copy()
 cap_removed=z['temporary_cap_faces_removed'].copy()
assert ipoints.dtype==np.float32 and dpoints.dtype==np.float32
assert np.all(np.isfinite(ipoints)) and np.all(np.isfinite(dpoints))
# The selected I asset is the open surface: its 41 temporary closure faces are
# recorded as removed, and no closure-cap source rows occur in its face table.
assert int(np.sum(isource==120000201))==0
edge_counts=collections.Counter()
for a,b,c in ifaces:
 for u,v in ((int(a),int(b)),(int(b),int(c)),(int(c),int(a))):edge_counts[tuple(sorted((u,v)))]+=1
open_edge_count=sum(v==1 for v in edge_counts.values())
assert open_edge_count==45
assert int(np.asarray(cap_removed).reshape(-1)[0])==41
# Verify the exact same native-current world frame against the pinned registered I CSV.
reg_xyz=[]
with I_REGISTERED.open(newline='') as f:
 for row in csv.DictReader(f):reg_xyz.append((float(row['wx']),float(row['wy']),float(row['wz'])))
reg_xyz=np.asarray(reg_xyz,dtype=np.float64)
assert reg_xyz.shape==ipoints.shape and np.array_equal(reg_xyz.astype(np.float32),ipoints)
# Reuse the pinned exact owner from the earlier 608 report; no seam groups are passed.
mod=loadmod('nearby_i_audit608_owner',HELP);pred=mod.load_exact_owner()
report608=json.loads(R608.read_text())
assert report608['pins']['baseline']==actual['baseline']
assert sha(pred.__file__)==report608['pins']['exact_owner']
# Build one I index in its own vertex/face namespace, so no shared IDs can
# suppress a real contact. D rows follow this independent source-ID selection.
i_raw,_=pred.float32_surface_records(ipoints.astype(np.float64),ifaces,source_seam_groups=())
i_face_base=-10_000_000_000_000
vertex_offset=len(dpoints)+17
irecs=[(r[0],r[1],r[2],i_face_base-j-1,tuple(vertex_offset+int(v) for v in r[4])) for j,r in enumerate(i_raw)]
assert len(irecs)==len(ifaces)
i_index=pred._prepare_surface_aabb(irecs)
d_raw,_=pred.float32_surface_records(dpoints.astype(np.float64),dfaces[new_rows],source_seam_groups=())
d_face_base=1_000_000_000_000
drecs=[(r[0],r[1],r[2],d_face_base+j,r[4]) for j,r in enumerate(d_raw)]
result=pred._audit_pair_prepared_first(i_index,drecs,same_surface=False)
pairs=np.asarray(result['triangle_pairs'],dtype=np.int64).reshape(-1,2)
# Preserve exact pair indices and emit every pair with provenance for both sides.
np.save(OUT/'fine-I-vs-changed-D-exact-pairs.npy',pairs,allow_pickle=False)
i_ends=-i_face_base-1
bytx=collections.Counter(); by_source=collections.Counter(); bybody=collections.Counter(); kinds=collections.Counter()
rows=[]
for k,(i_face,d_face) in enumerate(pairs):
 irow=int(-int(i_face)+i_face_base-1) if False else int((-int(i_face))-(-i_face_base))
 # i_face == i_face_base - row - 1
 irow=int(i_face_base-int(i_face)-1)
 j=int(d_face)-d_face_base; drow=int(new_rows[j]); sid=int(dsource[drow])
 tid=270 if sid==62050000 else 309 if sid==62050001 else (sid-62010000)//100
 tri_i=ifaces[irow]; tri_d=dfaces[drow]
 coords_i={tuple(map(float,ipoints[int(v)])) for v in tri_i}
 coords_d={tuple(map(float,dpoints[int(v)])) for v in tri_d}
 shared=len(coords_i&coords_d)
 # This is descriptive geometry only. It is not a documented-interface waiver.
 relation={3:'coincident_triangle_vertices',2:'shared_exact_edge_vertices',1:'shared_exact_vertex',0:'no_shared_vertex_coordinates'}[shared]
 ni=np.cross(ipoints[int(tri_i[1])].astype(np.float64)-ipoints[int(tri_i[0])].astype(np.float64),ipoints[int(tri_i[2])].astype(np.float64)-ipoints[int(tri_i[0])].astype(np.float64))
 nd=np.cross(dpoints[int(tri_d[1])].astype(np.float64)-dpoints[int(tri_d[0])].astype(np.float64),dpoints[int(tri_d[2])].astype(np.float64)-dpoints[int(tri_d[0])].astype(np.float64))
 denom=float(np.linalg.norm(ni)*np.linalg.norm(nd)); normal_dot=float(ni@nd/denom) if denom>0 else None
 # Direct provenance equality is recorded but never treated as sufficient to waive a pair.
 same_original_id=bool(int(isource[irow])==sid)
 same_face_identity=bool(int(iid0[irow])==int(did0[drow]) and int(iid1[irow])==int(did1[drow]))
 documented='not_proven' # no explicit I/D paired-interface certificate is part of either input
 bytx[tid]+=1;by_source[sid]+=1;bybody[(int(ibody[irow]),int(dbody[drow]))]+=1;kinds[relation]+=1
 rows.append({
  'pair_index':k,'transaction_id':tid,
  'D_candidate_face_row':drow,'D_source_original_id':sid,'D_source_face_id':int(dfaceid[drow]),'D_body':int(dbody[drow]),'D_layer':int(dlayer[drow]),'D_identity0':int(did0[drow]),'D_identity1':int(did1[drow]),
  'I_face_row':irow,'I_source_original_id':int(isource[irow]),'I_source_face_id':int(ifaceid[irow]),'I_body':int(ibody[irow]),'I_layer':int(ilayer[irow]),'I_identity0':int(iid0[irow]),'I_identity1':int(iid1[irow]),
  'shared_exact_float32_vertices':shared,'geometric_relation':relation,'normalized_face_normal_dot':normal_dot,
  'same_source_original_id':same_original_id,'same_identity_pair':same_face_identity,
  'documented_interface_status':documented,'waived':False,
 })
with (OUT/'fine-I-vs-changed-D-exact-pairs.csv').open('w',newline='') as f:
 fields=list(rows[0]) if rows else ['pair_index','transaction_id','D_candidate_face_row','D_source_original_id','D_source_face_id','D_body','D_layer','D_identity0','D_identity1','I_face_row','I_source_original_id','I_source_face_id','I_body','I_layer','I_identity0','I_identity1','shared_exact_float32_vertices','geometric_relation','normalized_face_normal_dot','same_source_original_id','same_identity_pair','documented_interface_status','waived']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
summary={
 'schema':'changed-candidate025-D-vs-accepted-fine-I-exact-structure-gate.v1',
 'status':'COMPLETE_EXACT_PAIR_QUERY',
 'scope':'all candidate025 faces whose source IDs are absent from baseline007 versus accepted open fine-I surface; no D/I interface exemption',
 'pins':{'candidate025_sha256':actual['candidate'],'baseline007_sha256':actual['baseline'],'fine_I_sha256':actual['fine_I'],'registered_I_csv_sha256':actual['registered_I'],'I_receipt_sha256':sha(I_RECEIPT),'I_frame_check':'registered-I.csv wx/wy/wz cast to float32 exactly equals I points_world_m row for row','exact_owner_sha256':sha(pred.__file__),'changed_candidate_face_rows_sha256':hashlib.sha256(new_rows.astype('<i8').tobytes()).hexdigest(),'predicate_owner_reference_report_sha256':sha(R608)},
 'selection':{'new_face_count':len(new_rows),'selection_rule':'all candidate face source IDs absent from baseline007','source_ids_absent_from_baseline007':True},
 'I_operand':{'face_count':len(ifaces),'vertex_count':len(ipoints),'open_boundary_edges':open_edge_count,'temporary_closure_faces_removed_recorded':int(cap_removed.reshape(-1)[0]),'closure_cap_source_rows_in_operand':int(np.sum(isource==120000201)),'prepared_I_index_count':1,'vertex_face_namespaces_disjoint':True},
 'query':{'exact_pair_count':int(result['count']),'aabb_candidate_pairs':int(result['aabb_candidate_pairs']),'same_surface':False,'source_seam_groups':[],'index_built_once':True},
 'pair_class_counts':dict(sorted(kinds.items())),
 'per_transaction':[{'transaction_id':int(k),'pairs':int(v)} for k,v in sorted(bytx.items())],
 'per_D_source_id':[{'source_id':int(k),'transaction_id':270 if k==62050000 else 309 if k==62050001 else (int(k)-62010000)//100,'pairs':int(v)} for k,v in sorted(by_source.items())],
 'per_body_pair':[{'I_body':int(a),'D_body':int(b),'pairs':int(v)} for (a,b),v in sorted(bybody.items())],
 'documented_interfaces':{'count':0,'status':'no explicit paired I/D interface witness is present in these inputs; all exact pairs remain unwaived and listed in CSV'},
 'outputs':{'pairs_npy':str(OUT/'fine-I-vs-changed-D-exact-pairs.npy'),'pairs_csv':str(OUT/'fine-I-vs-changed-D-exact-pairs.csv')},
 'physical_admission':False,'motion_qualification':False,'elapsed_s':time.monotonic()-started,
}
(OUT/'result.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':summary['status'],'pair_count':summary['query']['exact_pair_count'],'aabb_candidates':summary['query']['aabb_candidate_pairs'],'transactions':summary['per_transaction'],'classes':summary['pair_class_counts'],'out':str(OUT)},sort_keys=True),flush=True)
