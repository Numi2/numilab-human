"""Bind admitted source-only I topology into Skin's physical target; no geometry edits."""
from pathlib import Path
import json,numpy as np,hashlib,importlib.util
H=Path(__file__).resolve().parent
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
T=load('top358',H/'transfer_surface_contractions_to_volume_209_root.py')
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
out=H/'paired-I-physical-target-bind-358-root/skin-run001';out.mkdir(parents=True,exist_ok=False)
rp=H/'paired-inner-interface-quotient-352-root/run001/report.json';rr=json.loads(rp.read_text())
vr=next(x for x in rr['roles'] if x['role']=='Skin')
ip=Path(vr['input']);cp=H/'connected-skin-quality-356-root/skin-run001/cycle-00-edge/pass-004/fixed-boundary-volume-proposal.npz'
assert sha(cp)=='9d06d9c4088e71ba34e925fe7fe4d4f3a57ea32ea3da433fc12362bfac3b8a5d'
with np.load(ip) as z:oldp=z['points_world_m'];oldtarget=z['desired_boundary_faces'];oldt=z['tetrahedra']
assert np.array_equal(T.boundary_identity(oldtarget),T.boundary_identity(T.boundary(oldt)))
with np.load(cp) as z:a={k:z[k] for k in z.files}
p=a['points_world_m'];assert np.array_equal(oldp,p[:len(oldp)])
proof=H/'paired-inner-interface-quotient-353-root/typed-replay001/receipt.json'
assert json.loads(proof.read_text())['status']=='PASS_PAIRED_I_TYPED_ANCESTRY_AND_INDEPENDENT_VOLUME_REPLAY'
context=H/'paired-inner-interface-quotient-353-root/complete-context-audit354001/report.json'
assert sha(context)=='eaaff5dc170c187254f4fbe366c93e7fb5b3872916a807b94289c7e5450db00a'
with np.load(H/'paired-inner-interface-quotient-353-root/typed-replay001/candidate-I-typed-current.npz') as z:newI=z['points_world_m_f32'];newf=z['faces']
lookup={T.key(x):i for i,x in enumerate(p)};assert len(lookup)==len(p)
mp=np.arange(len(p))
for o in vr['operations']:mp[o['retired_volume_node']]=o['kept_volume_node']
desired=mp[oldtarget];degen=np.any(np.diff(np.sort(desired,axis=1),axis=1)==0,axis=1);desired=desired[~degen]
current=T.boundary(a['tetrahedra'])
assert np.array_equal(T.boundary_identity(current),T.boundary_identity(desired))
ifaces=np.asarray([[lookup[T.key(newI[v])] for v in f[::-1]] for f in newf],np.int64)
def keys(fs):return {tuple(x) for x in T.boundary_identity(fs)}
assert keys(ifaces).issubset(keys(desired))
assert len(desired)==len(oldtarget)-512 and len(ifaces)==249919
a.update(desired_boundary_faces=desired,boundary_faces=current,
 typed_I_candidate_path=np.asarray([str(H/'paired-inner-interface-quotient-353-root/typed-replay001/candidate-I-typed-current.npz')]),
 typed_I_candidate_sha256=np.asarray(['951f301ae896a3a5dc9c51b8e732b92a5cad38283ebda753dd9d4e1082c6cd18']))
pout=out/'skin-bound-current-I-target-volume-proposal.npz';np.savez_compressed(pout,**a)
with np.load(pout) as z:
 assert np.array_equal(z['points_world_m'],p) and np.array_equal(z['tetrahedra'],a['tetrahedra']) and np.array_equal(z['material_indices'],a['material_indices'])
r={'status':'PASS_EXACT_I_QUOTIENT_PHYSICAL_TARGET_BIND_ONLY','physical_admission':False,'native_steps':0,'input':str(cp),'input_sha256':sha(cp),'candidate':str(pout),'candidate_sha256':sha(pout),'typed_I_receipt':str(proof),'typed_I_receipt_sha256':sha(proof),'exact_context_report':str(context),'exact_context_report_sha256':sha(context),'desired_boundary_face_count':len(desired),'actual_boundary_equals_quotient_of_previously_verified_target':True,'complete_typed_I_faces_present_with_skin_orientation':True,'physical_arrays_bit_identical':True,'oriented_boundary_exactly_unchanged':True,'exact_F32_volume_unchanged':True,'producer_sha256':sha(Path(__file__)),'limitation':'Metadata binding only; Skin quality and full mass transfer remain incomplete.'}
(out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
fail=H/'connected-skin-quality-356-root/skin-run001/condition-target-guard-failure.json'
fail.write_text(json.dumps({'status':'RETAINED_STALE_DESIRED_TARGET_GUARD_FAILURE','log':str(fail.with_name('cycle-00-condition.log')),'reason':'328 correctly rejected obsolete pre-I-quotient desired_boundary_faces copied by raw352. 358 independently binds quotient of verified original target after exact source/volume proof. No failed geometry optimization occurred.','preserved_geometry_sha256':sha(cp),'physical_admission':False,'native_steps':0},indent=2)+'\n')

