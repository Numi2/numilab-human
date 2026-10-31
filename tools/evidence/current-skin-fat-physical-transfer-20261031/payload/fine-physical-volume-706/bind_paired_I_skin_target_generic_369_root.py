"""Rebind a proven paired I quotient to the complete Skin target, without geometry edits."""
from pathlib import Path
import json,numpy as np,hashlib,importlib.util,argparse,time
H=Path(__file__).resolve().parent
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
spec=importlib.util.spec_from_file_location('transfer369',H/'transfer_surface_contractions_to_volume_209_root.py');T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
ap=argparse.ArgumentParser()
for k in ['report','typed-receipt','context','output']:ap.add_argument('--'+k,type=Path,required=True)
args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
r=json.loads(args.report.read_text());assert r['status']=='PASS_PAIRED_I_QUOTIENT_PHYSICAL_PREFLIGHT'
pr=json.loads(args.typed_receipt.read_text());assert pr['status']=='PASS_PAIRED_I_TYPED_ANCESTRY_AND_INDEPENDENT_VOLUME_REPLAY' and pr['input_report_sha256']==sha(args.report)
ct=json.loads(args.context.read_text());assert ct['status']=='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
config=Path(ct['config']['path']);assert sha(config)==ct['config']['sha256'];cfg=json.loads(config.read_text())
typed=Path(pr['candidate']);assert sha(typed)==pr['candidate_sha256']==cfg['candidate_surfaces']['I']['sha256']
assert cfg['baseline_candidates']['I']['sha256']==r['source_I_sha256']
rr=next(x for x in r['roles'] if x['role']=='Skin')
source=Path(rr['input']);candidate=Path(rr['output']);assert sha(source)==rr['input_sha256'] and sha(candidate)==rr['output_sha256']
with np.load(source) as z:oldp=z['points_world_m'];oldtarget=z['desired_boundary_faces'];oldt=z['tetrahedra']
assert np.array_equal(T.boundary_identity(oldtarget),T.boundary_identity(T.boundary(oldt)))
with np.load(candidate) as z:a={k:z[k] for k in z.files}
p=a['points_world_m'];assert np.array_equal(oldp,p)
with np.load(typed) as z:ip=z['points_world_m_f32'];ff=z['faces']
lookup={T.key(x):i for i,x in enumerate(p)};assert len(lookup)==len(p)
mp=np.arange(len(p))
for o in rr['operations']:mp[o['retired_volume_node']]=o['kept_volume_node']
target=mp[oldtarget];degenerate=np.any(np.diff(np.sort(target,axis=1),axis=1)==0,axis=1);target=target[~degenerate]
current=T.boundary(a['tetrahedra']);assert np.array_equal(T.boundary_identity(current),T.boundary_identity(target))
ifaces=np.asarray([[lookup[T.key(ip[v])] for v in f[::-1]] for f in ff],np.int64)
keys=lambda f:{tuple(x) for x in T.boundary_identity(f)}
assert keys(ifaces).issubset(keys(target))
assert len(target)==len(oldtarget)-2*len(rr['operations'])
a.update(desired_boundary_faces=target,boundary_faces=current,typed_I_candidate_path=np.asarray([str(typed)]),typed_I_candidate_sha256=np.asarray([sha(typed)]))
outfile=args.output/'skin-bound-current-I-target-volume-proposal.npz';np.savez_compressed(outfile,**a)
with np.load(outfile) as z:
 for k,v in a.items():assert np.array_equal(z[k],v)
res={'status':'PASS_EXACT_I_QUOTIENT_PHYSICAL_TARGET_BIND_ONLY','physical_admission':False,'native_steps':0,'input':str(candidate),'input_sha256':sha(candidate),'candidate':str(outfile),'candidate_sha256':sha(outfile),'typed_I_receipt':str(args.typed_receipt),'typed_I_receipt_sha256':sha(args.typed_receipt),'exact_context_report':str(args.context),'exact_context_report_sha256':sha(args.context),'desired_boundary_face_count':len(target),'actual_boundary_equals_quotient_of_previously_verified_target':True,'complete_typed_I_faces_present_with_skin_orientation':True,'physical_arrays_bit_identical':True,'oriented_boundary_exactly_unchanged':True,'exact_F32_volume_unchanged':True,'native_quality_failures':rr['bad_after'],'producer_sha256':sha(Path(__file__)),'elapsed_s':time.monotonic()-start,'limitation':'Metadata binding only; all native quality and mass gates remain mandatory.'}
(args.output/'receipt.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res),flush=True)

