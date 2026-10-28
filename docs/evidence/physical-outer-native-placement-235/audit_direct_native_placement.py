from pathlib import Path
import json,hashlib,sys,importlib.util,time,signal
from fractions import Fraction
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
OUT=Path(__file__).resolve().parent
BASE=P/'root-connected-layer-fit-219/area-guard-connected-fit-008/matched-baseline-conditionedO-refinedID-attempt004/O.npz'
FRAME=P/'P/native-kind-aware-source-preflight-226/run-006-native-emission/skin-audit-001/cheap-gates.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
z=np.load(BASE,allow_pickle=False);d={k:z[k].copy() for k in z.files};a=json.loads(FRAME.read_text())['common_placement'];R=np.array(a['rotation_row_major']).reshape(3,3);t=np.array(a['translation_m'])
old=np.asarray(d['points_world_m'],np.float32);p=(old.astype(float)@R.T+t).astype(np.float32);f=np.asarray(d['source_O_faces'],np.int64)
d['points_world_m']=p;d['native_placement_rotation_row_major']=R.ravel();d['native_placement_translation_m']=t;d['preplacement_source_sha256']=np.array(sha(BASE));np.savez_compressed(OUT/'O-direct-native-placement.npz',**d)
tri=p.astype(float)[f];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);oldtri=old.astype(float)[f];on=np.cross(oldtri[:,1]-oldtri[:,0],oldtri[:,2]-oldtri[:,0])@R.T
area=np.linalg.norm(n,axis=1);oa=np.linalg.norm(on,axis=1);co=np.einsum('ij,ij->i',n,on)/(area*oa)
r={'input':str(BASE),'input_sha256':sha(BASE),'placement_source':str(FRAME),'placement_source_sha256':sha(FRAME),'output':str(OUT/'O-direct-native-placement.npz'),'output_sha256':sha(OUT/'O-direct-native-placement.npz'),'points':len(p),'faces':len(f),'finite':bool(np.isfinite(p).all()),'degenerate':int(np.sum(area==0)),'flipped':int(np.sum(co<=0)),'minimum_normal_cosine':float(co.min()),'placement':a,'physical_steps':0,'admitted':False}
(OUT/'preflight.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
if r['degenerate'] or r['flipped']:raise SystemExit(2)
tool=P.parent/'boolean-prototype-001/audit_self_envelope006.py'
spec=importlib.util.spec_from_file_location('direct_native_exact_owner',tool);audit=importlib.util.module_from_spec(spec);sys.modules[spec.name]=audit;spec.loader.exec_module(audit);pred=audit.load_exact();start=time.monotonic()
def alarm(s,f):raise TimeoutError('180second exact audit budget exceeded')
signal.signal(signal.SIGALRM,alarm);signal.alarm(180)
try:
 recs=pred._records(tuple(tuple(Fraction.from_float(float(v)) for v in x) for x in p),tuple(tuple(map(int,x)) for x in f));result=pred._audit_pair(recs,recs,same_surface=True)
 r|={'status':'EXACT_DIRECT_PLACEMENT_SELF_COMPLETE','exact_predicate':str(audit.EXACT_OWNER),'exact_predicate_sha256':sha(audit.EXACT_OWNER),'exact_self_pairs':int(result['count']),'triangle_pairs':result['triangle_pairs'],'aabb_candidate_pairs':int(result['aabb_candidate_pairs']),'elapsed_seconds':time.monotonic()-start};(OUT/'report.json').write_text(json.dumps(r,indent=2)+'\n');np.save(OUT/'self-pairs.npy',np.asarray(result['triangle_pairs'],np.int64).reshape(-1,2));print(json.dumps({k:v for k,v in r.items() if k!='triangle_pairs'}),flush=True)
finally:signal.alarm(0)
