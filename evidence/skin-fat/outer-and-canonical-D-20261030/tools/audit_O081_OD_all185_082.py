from pathlib import Path
import hashlib, importlib.util, json, sys, time
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
AUDIT= P/'complete-cross-surface-pair-audit-654/audit_cross_layer_pairs_654.py'
OUT=P/'reduced-physical-boundary-400/attempt-046-fine-role-simplification/run-082-O081-D-all185-gates'
O=P/'reduced-physical-boundary-400/attempt-046-fine-role-simplification/run-081-OI-local-topology-repair/candidate-O-self-and-OI-repaired.npz'
O_REPORT=P/'reduced-physical-boundary-400/attempt-046-fine-role-simplification/run-081-OI-local-topology-repair/report.json'
OI_REPORT=O_REPORT
I=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/dnew-vs-i296-inventory-001/i328-cavity-audit-001/fill-003/I-f32-conditioning-367/condition-288-pair-connected-remesh-001-run-002/result/candidate-f32-typed.npz'
D=P/'canonical-D-boundary-binding-696/run-007-collar-expanded-safe-replay/source-D-canonical14-stars-conditioned.npz'
D_GATES=D.parent/'exact-gates.json'
BONES=P/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz'
EXPECTED={'O':'c9074b9f54bfad16193cfd14dc52fbdcda0fcca86f01a34f9552529fc6bb6b72','O_report':'da8d23dadac4bccacca20be69750a78b67352a08b69694d455d9ab3a35a53ce3','I328':'5d46ee99df2613e9f30a43075af95a6402beffe26392cd9f017d8c1a154bf410','D007':'3b3a5c183b49504aa00a4630cbc71b0bbecb0d5618dc8bbe958b5fd259971841','bones':'e9f0fffefa68f274e8d1c0a565d3d6cab1cfba31d13e085b8fdc79ca362185e5','audit_script':'dc5c1c58a07f5ac28e2f41eb14319c76a17326e21bb710aa99ffdac71a84c187'}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def writej(p,obj):
 q=Path(p).with_suffix(Path(p).suffix+'.tmp');q.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n');q.replace(p)
def load_module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
if OUT.exists():raise FileExistsError(OUT)
OUT.mkdir()
START=time.monotonic()
REPORT={'schema':'numi.O081-D-and-all185-exact-gates.082','status':'PREPARING','physical_admission':False,'runtime_created':0,'accepted_physical_steps':0,'input_pins':{},'completed_gates':[],'failed_gate':None}
for n,p,key in [('O',O,'O'),('O_report',O_REPORT,'O_report'),('I328',I,'I328'),('D007',D,'D007'),('bones',BONES,'bones'),('audit_script',AUDIT,'audit_script')]:
 h=sha(p)
 if key and h!=EXPECTED[key]:raise RuntimeError(f'{n} SHA mismatch: {h}')
 REPORT['input_pins'][n]={'path':str(p),'sha256':h}
# Bind the prior O self + exhaustive O-I gates from the exact candidate report.
selfr=json.loads(O_REPORT.read_text())
if selfr.get('final',{}).get('O_self_pair_count')!=0:raise RuntimeError('O081 candidate self receipt is not zero')
oic=selfr.get('full_cross', {})
if selfr.get('status')!='PASS_O_SELF_AND_EXHAUSTIVE_OI_ZERO' or oic.get('count')!=0 or not oic.get('complete'):raise RuntimeError('O081 exhaustive O-I receipt is not zero/complete')
REPORT['completed_gates'].append({'gate':'O_self','status':'PASS','exact_candidate_self_pairs':0,'receipt_path':str(O_REPORT),'receipt_sha256':sha(O_REPORT)})
REPORT['completed_gates'].append({'gate':'O-I','status':'PASS_ZERO_EXACT_PAIRS','exact_pair_count':0,'aabb_candidate_pairs':oic.get('aabb_candidates'),'aabb_scanned':oic.get('scanned'),'receipt_path':str(OI_REPORT),'receipt_sha256':sha(OI_REPORT)})
# Reuse the pinned 654 exact loader and pair-audit owner. Override only the O artifact path and output directory.
audit=load_module(AUDIT,'audit654_run074')
audit.O_PATH=O
audit.D_PATH=D
audit.OUT=OUT
audit.START=START
loader=load_module(audit.LOADER,'loader654_run074')
pred=loader.load_exact_owner()
if sha(Path(pred.__file__))!=audit.EXACT_OWNER_SHA:raise RuntimeError('exact geometry owner SHA mismatch')
if sha(audit.LOADER)!=audit.LOADER_SHA:raise RuntimeError('loader SHA mismatch')
REPORT['input_pins']['exact_owner']={'path':str(pred.__file__),'sha256':sha(Path(pred.__file__))}
REPORT['input_pins']['loader']={'path':str(audit.LOADER),'sha256':sha(audit.LOADER)}
REPORT['input_pins']['driver']={'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__).resolve())}
writej(OUT/'progress.json',REPORT)
# O-I first, using the unwarped accepted I328 surface.
def run_cross(name):
 ar,br,info,ap,af,bp,bf=audit.build_records(pred,name)
 if len(ar)!=af or len(br)!=bf:raise RuntimeError(f'{name} record construction dropped a face')
 count=audit.count_candidates(pred,ar,br)
 pre={'A':info['A'],'B':info['B'],'A_point_rows':ap,'A_faces':af,'B_point_rows':bp,'B_faces':bf,'aabb_candidate_pairs':int(count),'all_aabb_candidates_counted_before_exact':True}
 writej(OUT/('preflight-'+name.replace('-','_')+'.json'),pre)
 result=audit.audit_pair(pred,name,pre)
 return result
for name in ('O-D',):
 result=run_cross(name)
 REPORT['completed_gates'].append({'gate':name,'status':result['status'],'exact_pair_count':result['exact_pair_count'],'aabb_candidate_pairs':result['aabb_candidate_pairs_total'],'aabb_scanned':result['aabb_candidate_pairs_exactly_scanned'],'witness':result['first_witness']})
 writej(OUT/'progress.json',REPORT)
 print(json.dumps({'gate':name,'status':result['status'],'exact_pairs':result['exact_pair_count'],'aabb':result['aabb_candidate_pairs_total'],'witness':result['first_witness']},sort_keys=True),flush=True)
 if result['exact_pair_count']:
  REPORT['status']='FAILED_'+name.replace('-','_');REPORT['failed_gate']=REPORT['completed_gates'][-1]
  writej(OUT/'report.json',REPORT);raise SystemExit(2)
# All 185 native bone operands; query the exact full emitted surface records with no pair exclusions.
with np.load(O,allow_pickle=False) as z:op=np.asarray(z['points_world_m'],np.float32).astype(np.float64);of=np.asarray(z['faces'],np.int64)
with np.load(BONES,allow_pickle=False) as z:bp=np.asarray(z['points_world_f32'],np.float32).astype(np.float64);bf=np.asarray(z['faces'],np.int64);bo=np.asarray(z['bank_face_offsets'],np.int64);bb=np.asarray(z['bank_body'],np.int64);bn=np.asarray(z['bank_bone'],np.int64);bs=np.asarray(z['bank_source_id'],np.int64)
if (len(of),len(bf),len(bo)-1)!=(76937,469136,185):raise RuntimeError('all185 input dimensions changed')
orows,_=pred.float32_surface_records(op,of,source_seam_groups=())
brows,_=pred.float32_surface_records(bp,bf,source_seam_groups=())
if len(orows)!=len(of) or len(brows)!=len(bf):raise RuntimeError('all185 face-record construction dropped a face')
res=pred._audit_pair(orows,brows,same_surface=False)
pairs=np.asarray(res.get('triangle_pairs',[]),np.int64).reshape(-1,2)
if len(pairs):
 bankrow=np.searchsorted(bo[1:],pairs[:,1],side='right')
 by=[]
 for i in sorted(set(map(int,bankrow))):
  sub=pairs[bankrow==i]
  by.append({'bank_array_row':i,'body':int(bb[i]),'bone':int(bn[i]),'source_id':int(bs[i]),'pair_count':int(len(sub)),'O_face_rows':sorted(set(map(int,sub[:,0]))),'bank_face_rows':sorted(set(map(int,sub[:,1]-bo[i])))})
 first=next((x for x in pairs if int(x[0]) in {int(r[3]) for r in orows} and int(x[1]) in {int(r[3]) for r in brows}),pairs[0])
 omap={int(r[3]):r for r in orows};bmap={int(r[3]):r for r in brows}
 witness=audit.first_exact_crossing(pred,omap[int(first[0])],bmap[int(first[1])])
else:bankrow=np.empty((0,),np.int64);by=[];witness=None
np.savez_compressed(OUT/'O-vs-all185-native-bones-witnesses.npz',triangle_pairs=pairs,bank_array_row_by_pair=bankrow,bank_face_offsets=bo,bank_body=bb,bank_bone=bn,bank_source_id=bs)
status='PASS_ZERO_EXACT_PAIRS' if len(pairs)==0 else 'FAILED_FIRST_EXACT_CROSSING'
REPORT['completed_gates'].append({'gate':'O-all185-native-bones','status':status,'exact_pair_count':int(len(pairs)),'aabb_candidate_pairs':int(res.get('aabb_candidate_pairs',0)),'bank_count':185,'bank_pair_counts':by,'first_witness':witness,'scope':'full O081 emitted surface against all 185 native F32 bone operands; distinct surfaces, no shared-feature waiver'})
REPORT['failed_gate']=None if not len(pairs) else REPORT['completed_gates'][-1]
REPORT['status']='PASS_O081_D_AND_ALL185_GATES' if not len(pairs) else 'FAILED_O_ALL185_NATIVE_BONES'
REPORT['elapsed_s']=time.monotonic()-START;REPORT['physical_admission']=False;REPORT['runtime_created']=0;REPORT['accepted_physical_steps']=0
writej(OUT/'report.json',REPORT)
print(json.dumps({'status':REPORT['status'],'O_self':0,'gates':REPORT['completed_gates'],'elapsed_s':REPORT['elapsed_s'],'report':str(OUT/'report.json')},sort_keys=True),flush=True)
if REPORT['status']!='PASS_O081_D_AND_ALL185_GATES':raise SystemExit(2)
