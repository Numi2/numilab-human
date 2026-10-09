from __future__ import annotations
import hashlib, importlib.util, json, os, struct, sys, time
from pathlib import Path
import numpy as np

ROOT=Path('/Users/n/numi-human-retained-delivery-20261009')
RUN_DIR=ROOT/'passive-biceps-micro-overlap-1225/native-biceps-10s-attempt001/native-run'
CAND=ROOT/'passive-biceps-micro-overlap-1225/compose-current-1cd-attempt003'
REVIEW=ROOT/'passive-biceps-micro-overlap-1225/native-biceps-10s-attempt001/review-003'
TISS=CAND/'bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
MANIFEST=CAND/'bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json'
ANATOMY_RECEIPT=CAND/'resting-anatomy-receipt.json'
COMPOSE_REPORT=CAND/'report.json'
COMPOSE_EXEC=CAND/'execution.json'
HIST_SRC=Path('/Users/n/numi-human-common-skin-multipose-001/src')
CI=HIST_SRC/'numilab_human/cardiac_cavity_intersections.py'
CLEARANCE=HIST_SRC/'numilab_human/common_atlas_skin_clearance.py'
AUDIT_1213=ROOT/'passive-muscle-self-audit-1213/audit_muscle_self_1213.py'
RUN_META=RUN_DIR/'run-metadata.json'
INVOCATION=RUN_DIR/'invocation.json'
STEPS=(0,5000)
SURFACE_IDS=(103,104)


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()

def require(ok,msg):
 if not ok: raise RuntimeError(msg)

def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 require(spec is not None and spec.loader is not None,f'cannot load {path}')
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ[name]='1'
if REVIEW.exists(): raise FileExistsError(REVIEW)
REVIEW.mkdir(parents=True)

# Exact copied predicates/loaders used by the retained 1213 quotient audit.
ci_expected='11f042cc4809a27735c02b66a8fb07ebc92ee66597d7e1814a75231e2cf2d2bb'
clear_expected='fbfb5dfbb9b3f7e5fa297a61c793c66cb13fe884d5cebc3a7d1a6b9908961ac1'
audit_expected=''
for p,expected in [(CI,ci_expected),(CLEARANCE,clear_expected)]: require(sha(p)==expected,f'predicate/loader pin mismatch: {p}')
sys.path.insert(0,str(HIST_SRC))
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human import common_atlas_skin_clearance as clearance
auditmod=load_module('_biceps_quotient_contract_1213',AUDIT_1213)
require(Path(ci.__file__).resolve()==CI.resolve(),'predicate imported from an unexpected path')
require(Path(clearance.__file__).resolve()==CLEARANCE.resolve(),'pack loader imported from an unexpected path')

manifest=json.loads(MANIFEST.read_text())
receipt=json.loads(ANATOMY_RECEIPT.read_text())
compose_exec=json.loads(COMPOSE_EXEC.read_text())
require(manifest['payload']['sha256']==sha(TISS) and manifest['payload']['file']==TISS.name,'NHTISS manifest payload pin mismatch')
require(manifest['payload']['vertex_count']==433151 and manifest['payload']['index_count']==1884573,'unexpected NHTISS dimensions')
require(receipt['provenance']['native_muscle_surfaces']['sha256']==sha(TISS),'receipt does not bind candidate NHTISS')
require(receipt['provenance']['native_muscle_surfaces']['manifest_sha256']==sha(MANIFEST),'receipt does not bind candidate manifest')
require(compose_exec['accepted_pose_forward_status']=='not_run','composition report incorrectly claims pose-forward validation')
require(compose_exec['changed_stable_ids']==[103,104] and compose_exec['binding_table_byte_exact'] and compose_exec['other_row_vertex_bytes_match_current_parent'],'composition lineage checks do not match the declared two rows')
meta=json.loads(RUN_META.read_text()); inv=json.loads(INVOCATION.read_text())
argv=meta.get('argv'); assets=meta.get('asset_sha256',{})
require(meta.get('exit_code')==0 and meta.get('loaded_metal_runtime',{}).get('verified') is True,'native run metadata does not establish a successful verified runtime')
require(argv==inv.get('argv') and assets==inv.get('asset_sha256'),'native invocation and metadata differ')
require(not meta.get('source_files_changed_during_run'),'source changed during native run')
require('--soft-tissue-payload' in argv and argv[argv.index('--soft-tissue-payload')+1]==str(TISS),'native invocation did not load exact candidate NHTISS')
require('--resting-anatomy-receipt' in argv and argv[argv.index('--resting-anatomy-receipt')+1]==str(ANATOMY_RECEIPT),'native invocation did not load exact candidate receipt')
require('--muscle-step-count' in argv and argv[argv.index('--muscle-step-count')+1]=='5000','native invocation is not the prepared 5,000-step run')
require(assets.get(str(TISS))==sha(TISS),'runtime asset map does not bind exact candidate NHTISS')
require(assets.get(str(ANATOMY_RECEIPT))==sha(ANATOMY_RECEIPT),'runtime asset map does not bind exact candidate receipt')

raw=TISS.read_bytes()
magic,abi,nrecords,nbindings,nvertices,nindices,fingerprint,archive_sha=struct.unpack_from('<8s6I32s',raw,0)
require(magic==b'NHTISS4\0' and abi==5 and nrecords==150 and nvertices==433151 and nindices==1884573,'unsupported/mismatched NHTISS4 header')
require(len(raw)==64+32*nrecords+36*nbindings+56*nvertices+4*nindices,'NHTISS payload byte-range mismatch')
records=np.frombuffer(raw,dtype='<u4',count=nrecords*8,offset=64).reshape(-1,8)
vertex_offset=64+32*nrecords+36*nbindings
source_positions=np.ndarray((nvertices,3),dtype='<f4',buffer=raw,offset=vertex_offset,strides=(56,4)).copy()
index_offset=vertex_offset+56*nvertices
indices=np.frombuffer(raw,dtype='<u4',count=nindices,offset=index_offset).copy()
rows={int(r.get('stable_id')):r for r in manifest['source']['surfaces'] if r.get('layer')=='muscle'}
records_by_id={int(r[6]):r for r in records if int(r[7])==1}
require(set(SURFACE_IDS).issubset(rows) and set(SURFACE_IDS).issubset(records_by_id),'requested stable IDs not present as NHTISS muscle surfaces')

steps_data={}
for step in STEPS:
 pack=RUN_DIR/'accepted-geometry'/f'step-{step}.mrvpack'
 prec=RUN_DIR/'accepted-geometry'/f'step-{step}.receipt.json'
 pd=json.loads(prec.read_text())
 require(pd.get('accepted_step')==step and pd.get('physical_endpoint')=='accepted','capture receipt is not the requested accepted endpoint')
 require(pd.get('pack_file_sha256')==sha(pack),'capture receipt pack SHA mismatch')
 require(Path(pd.get('accepted_pack_path','')).resolve()==pack.resolve(),'capture receipt points to another pack')
 ptime=float(pd['accepted_time_s'])
 require(abs(ptime-(step*float(argv[argv.index('--muscle-step-seconds')+1])))<2e-6,'accepted capture clock does not match physical dt')
 keys={(51005,sid) for sid in SURFACE_IDS}
 positions,surfaces,counts=clearance._pack_surfaces(pack,keys)
 require(keys.issubset(surfaces),'capture lacks one or more requested NHTISS primitive IDs')
 require({k for k in surfaces if k[0]==51005}==keys,'requested muscle key subset is not exact for stable IDs 103 and 104')
 steps_data[step]={'pack':pack,'receipt':prec,'positions':positions,'surfaces':surfaces,'counts':counts,'receipt_doc':pd,'accepted_time_s':ptime}

results=[]
start=time.monotonic()
for sid in SURFACE_IDS:
 mr=rows[sid]; rec=records_by_id[sid]
 first_binding,binding_count,first_vertex,vertex_count,first_index,index_count,stable_id,layer_code=map(int,rec)
 require(stable_id==sid and layer_code==1 and index_count%3==0,'NHTISS muscle record header mismatch')
 source_faces=(indices[first_index:first_index+index_count].reshape(-1,3)-first_vertex).astype(np.int64)
 used=np.unique(source_faces)
 faces=np.searchsorted(used,source_faces)
 source_xyz=source_positions[first_vertex+used].copy()
 entry={'stable_id':sid,'member_id':mr.get('member_id'),'label':mr.get('label'),'body_bindings':mr.get('body_bindings',[]),'nhtiss_vertex_count':vertex_count,'face_count':len(source_faces),'source_nhtiss_face_rows_preserved':True,'coordinate_spaces':[]}
 source_result=auditmod.audit_space(ci,source_xyz,faces,used,vertex_count,'authored_candidate_nhtiss4_owner_local_source')
 entry['coordinate_spaces'].append(source_result)
 for step in STEPS:
  item=steps_data[step]
  captured=item['surfaces'][(51005,sid)]['faces']
  delta=captured-source_faces
  require(len(captured)==len(source_faces) and delta.size>0 and np.all(delta==delta.flat[0]),f'captured face rows do not map one-to-one to source NHTISS for stable_id {sid} step {step}')
  pack_offset=int(delta.flat[0])
  require(np.array_equal(captured,pack_offset+source_faces),f'captured face order/index identity mismatch for stable_id {sid} step {step}')
  posed_xyz=item['positions'][pack_offset+used].copy()
  r=auditmod.audit_space(ci,posed_xyz,faces,used,vertex_count,f'accepted_mrvpack2_step_{step}_world')
  r['accepted_step']=step;r['accepted_time_s']=item['accepted_time_s'];r['pack_sha256']=sha(item['pack']);r['receipt_sha256']=sha(item['receipt']);r['pack_vertex_offset']=pack_offset;r['captured_face_rows_exactly_map_to_nhtiss']=True
  entry['coordinate_spaces'].append(r)
 results.append(entry)

ledger=REVIEW/'unallowed-self-pairs.jsonl'
with ledger.open('x') as f:
 for surface in results:
  for space in surface['coordinate_spaces']:
   for w in space['unallowed_pair_witnesses']:
    f.write(json.dumps({'stable_id':surface['stable_id'],'member_id':surface['member_id'],'label':surface['label'],'coordinate_space':space['coordinate_space'],**{k:v for k,v in w.items() if k!='triangle_xyz_f32_m'},'triangle_xyz_f32_m':w['triangle_xyz_f32_m']},separators=(',',':'),allow_nan=False)+'\n')
report={
 'schema':'numi.human.biceps-103-104-exact-self-audit.v1',
 'status':'completed',
 'qualification':'Exact Float32 coordinate quotient and same-surface intersection predicate for NHTISS surfaces 103 and 104 only, at authored source plus actual accepted steps 0 and 5000. Not a whole-muscle cross-surface, skin-target, full-body anatomy, or clinical qualification.',
 'native_run':{'directory':str(RUN_DIR),'declaration_sha256':'e7e716506ab85d81c5e74fc26e99447861d3f3712605f9c4bdde0ac0a2b0d4e5','execution_sha256':sha(RUN_DIR.parent/'execution.json'),'run_metadata_sha256':sha(RUN_META),'invocation_sha256':sha(INVOCATION),'exit_code':meta['exit_code'],'loaded_runtime_verified':meta['loaded_metal_runtime']['verified'],'source_files_changed_during_run':meta.get('source_files_changed_during_run')},
 'anatomy_inputs':{'nhtiss_payload':{'path':str(TISS),'sha256':sha(TISS),'abi':abi,'surface_count':nrecords,'vertex_count':nvertices,'index_count':nindices},'manifest':{'path':str(MANIFEST),'sha256':sha(MANIFEST)},'receipt':{'path':str(ANATOMY_RECEIPT),'sha256':sha(ANATOMY_RECEIPT),'parent_1218_receipt_sha256':'ebfb61b926f33dd6497176734324e7208e1eeab2f2b518e6d4f884e0a8da99ec'},'composition_report':{'path':str(COMPOSE_REPORT),'sha256':sha(COMPOSE_REPORT),'changed_stable_ids':[103,104],'accepted_pose_forward_status':'not_run'}},
 'predicate':{'method':'exact per-surface Float32 coordinate quotient, then historical cardiac_cavity_intersections._audit_pair(records, records, same_surface=True); no tolerance weld, movement, face edits, or predicate change','audit_contract_source':{'path':str(AUDIT_1213),'sha256':sha(AUDIT_1213)},'predicate_source':{'path':str(CI),'sha256':sha(CI)},'pack_loader_source':{'path':str(CLEARANCE),'sha256':sha(CLEARANCE)}},
 'capture_inputs':[{'step':s,'accepted_time_s':steps_data[s]['accepted_time_s'],'pack_path':str(steps_data[s]['pack']),'pack_sha256':sha(steps_data[s]['pack']),'receipt_path':str(steps_data[s]['receipt']),'receipt_sha256':sha(steps_data[s]['receipt']),'face_rows_match_source':True} for s in STEPS],
 'surfaces':results,
 'pair_ledger':{'path':str(ledger),'sha256':sha(ledger),'rows':sum(sp['self_intersection']['unallowed_pair_count'] for s in results for sp in s['coordinate_spaces'])},
 'wall_seconds':time.monotonic()-start,
}
report_path=REVIEW/'report.json';report_path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'report':str(report_path),'report_sha256':sha(report_path),'pair_ledger_sha256':sha(ledger),'per_surface':[{ 'stable_id':s['stable_id'],'label':s['label'],'spaces':[{'name':x['coordinate_space'],'unallowed':x['self_intersection']['unallowed_pair_count'],'degenerate':x['self_intersection']['omitted_degenerate_face_count'],'closed_oriented':x['topology']['closed_oriented_manifold_candidate'],'candidate_pairs':x['self_intersection']['aabb_candidate_pairs']} for x in s['coordinate_spaces']]} for s in results],'wall_seconds':report['wall_seconds']},indent=2,sort_keys=True),flush=True)
