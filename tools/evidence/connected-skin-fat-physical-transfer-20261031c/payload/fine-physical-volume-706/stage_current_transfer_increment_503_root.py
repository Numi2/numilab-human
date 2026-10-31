
from pathlib import Path
import hashlib,json,subprocess,os,time
H=Path(__file__).resolve().parent;P=H.parent;PUB=Path('/Users/numi/Work/human-volume-proof-publication-20261031')
PARENT='fe7d08599b12fcbbaff823907c71f5376a081350'
REL='tools/evidence/connected-skin-fat-physical-transfer-20261031c'
PKG=PUB/REL;OUT=H/'publication-503-root'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=PUB,text=True).strip()
assert git('rev-parse','HEAD')==PARENT and not git('status','--porcelain')
cp=H/'checkpoint-current502-root.json';j=json.loads(cp.read_text());state=j['current_verified_state_20261031']
assert sha(cp)=='73fd3c5bba68c3e82f6d30c573def32a5a60937cdd2e274c87b11a6e8650b275'
assert not state['physical_admission'] and state['native_steps']==0
proofs={
 'full_joint_context':H/'joint-current-source-497-root/receipt.json',
 'D475_volume_independent':H/'d-shortedge-quotient-475-next-001-preflight-100um/volume-transfer-001/independent-replay-001/report.json',
 'Fat_fixed_boundary_LP':H/'fixed-fat-coordinate-views-496-root/independent495001/receipt.json',
 'Skin_I_flip_volume':H/'skin-I-flip-independent-501-root/receipt.json',
 'Skin_O_quotient':H/'outer-O-quality-exchange-484-root/exchange/qualification435001/receipt.json'}
M=P/'fractional-semantic-support-090-sparse-graph/fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001'
mass=M/'attempt-022-source-exhaustion-reserved-batch/independent-replay-013.json';proofs['mass022_partial_current406']=mass
for k,p in proofs.items():
 r=json.loads(p.read_text());assert r['status'].startswith('PASS') and not r.get('physical_admission',False) and r.get('native_steps',0)==0,(k,r['status'])
dirs=[
 'paired-I-quality-exchange-464-root','paired-I-quality-exchange-465-root','paired-I-quality-exchange-466-root','paired-I-cross-tissue-473-root',
 'combined-D414-I466-volume-468-root','combined-D414-I466-volume-469-root','joint-D414-I466-target-475-root',
 'd-shortedge-quotient-414-next-001-input-attempt004','d-shortedge-quotient-414-next-001-preflight-attempt004','d-shortedge-quotient-414-batch001-volume-transfer',
 'd-shortedge-quotient-475-next-001-preflight-100um',
 'outer-O-quality-exchange-484-root','fixed-fat-interior-LP-488-root','fixed-fat-identity-LP-492-root','fixed-fat-coordinate-views-496-root',
 'joint-current-source-497-root','skin-paired-I-flip-target-498-root','skin-I-flip-independent-501-root']
dirs += ['fat-comp0-connected-warning-repair-412/'+n for n in [
 'run-017-current-comp0-remesh','run-019-expanded-shell','run-020-link-closure-shell','run-023-D414-typed-warning-flips',
 'run-025-current-target-475-shell','run-026-current-target-475-prep','run-027-current-target-475-remesh',
 'run-028-warning-flips','run-030-warning-flips-current-context','run-031-threeD-oneI-safe-flips','run-032-oldbranch-control','run-033-target-rebind-480']]
endpoints={Path(state[k]['path']) for k in ('Skin_current','Fat_current')}
endpoints.update(Path(x['path']) for x in state['source_surfaces'].values())
endpoints.update([
 H/'joint-D414-I466-target-475-root/fat-target397/fat-current-typed-target-rebound.npz',
 H/'paired-I-quality-exchange-466-root/qualification467001/skin-target369/skin-bound-current-I-target-volume-proposal.npz',
 H/'paired-I-quality-exchange-466-root/typed463001/candidate-I-typed-current.npz',
 H/'d-shortedge-quotient-414-batch001-volume-transfer/run002/fat-D414-surface-quotient-volume-proposal.npz',
 H/'d-shortedge-quotient-475-next-001-preflight-100um/volume-transfer-001/run001/fat-D475-surface-quotient-volume-proposal.npz',
 H/'d-shortedge-quotient-475-next-001-preflight-100um/volume-transfer-001/typed-bind-002/candidate-D-typed-current.npz',
 H/'fixed-fat-identity-LP-492-root/run001/conditioned-volume-proposal.npz',
 H/'fixed-fat-coordinate-views-496-root/fat-all-coordinate-views-volume-proposal.npz',
 H/'outer-O-quality-exchange-484-root/exchange/qualification435001/skin-target437/skin-bound-current-O-target-volume-proposal.npz',
 H/'skin-paired-I-flip-target-498-root/skin-current-I-flip-target-bound.npz',
 H/'skin-paired-I-flip-target-498-root/remesh499-S0/connected-fixed-surface-volume-proposal.npz'])
sources={cp,H/'checkpoint-before-current502-root.json',Path(__file__)};omitted=[];oversized=[];missing_dirs=[]
for rel in dirs:
 d=H/rel
 if not d.is_dir():missing_dirs.append(str(d));continue
 for p in d.rglob('*'):
  if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts or p.suffix in ('.pyc','.tmp'):continue
  if p.suffix=='.npz' and p.stat().st_size>2_000_000 and p not in endpoints:
   omitted.append({'path':str(p),'sha256':sha(p),'size_bytes':p.stat().st_size,'status':'RETAINED_HOST_INTERMEDIATE_NOT_DUPLICATED'});continue
  if p.stat().st_size<50_000_000:sources.add(p)
for n in range(461,503):sources.update(H.glob('*'+str(n)+'*root.py'))
sources.update(proofs.values());sources.update(endpoints)
# Freeze each accepted mass batch, its controls and historical rejected source trials.
for rel in ['attempt-012-centroid-partial-batch','attempt-014-noncentroid-partial-batch','attempt-021-source-exhaustion-batch','attempt-022-source-exhaustion-reserved-batch']:
 for p in (M/rel).rglob('*'):
  if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.stat().st_size<90_000_000:sources.add(p)
mr=json.loads(mass.read_text())
for path,want in mr['inputs'].items():
 p=Path(path);assert sha(p)==want,(path,want)
 if p.stat().st_size<=90_000_000:sources.add(p)
 else:oversized.append({'path':path,'sha256':want,'size_bytes':p.stat().st_size,'status':'HOST_DEPENDENCY_TOO_LARGE_FOR_GITHUB_ORDINARY_BLOB_REQUIRES_EXISTING_RELEASE_ARCHIVE'})
# Current LP source masks and their already authenticated identity-coordinate source.
ir=H/'fat-inferred-identity-mask-336-root/report.json';identity=json.loads(ir.read_text())
sources.update([ir,H/'fat-inferred-identity-mask-336-root/identity-and-pending-cavity-mask.npz',Path(identity['source_input'])])
plan=[];syntax=[]
for p in sorted(sources,key=str):
 assert p.is_file() and p.is_relative_to(P) and not p.is_symlink(),p
 if p.stat().st_size>90_000_000:raise ValueError('unhandled ordinary Git blob size '+str(p))
 if p.suffix=='.py':
  try:compile(p.read_text(),str(p),'exec')
  except SyntaxError as ex:syntax.append({'path':str(p),'sha256':sha(p),'error':str(ex),'status':'HISTORICAL_FAILED_SOURCE_UNQUALIFIED'})
 plan.append({'source_path':str(p),'payload':str(Path('payload')/p.relative_to(P)),'sha256':sha(p),'size_bytes':p.stat().st_size})
OUT.mkdir(exist_ok=False);PKG.mkdir(exist_ok=False,parents=True);start=time.monotonic();v=os.statvfs(H);before=v.f_bavail*v.f_frsize
for e in plan:
 src=Path(e['source_path']);dest=PKG/e['payload'];dest.parent.mkdir(exist_ok=True,parents=True)
 assert sha(src)==e['sha256'];subprocess.run(['/bin/cp','-c','-p',str(src),str(dest)],check=True)
 assert sha(dest)==e['sha256'] and dest.stat().st_size==e['size_bytes'] and sha(src)==e['sha256']
manifest={'schema':'numi-human.evidence-payload.v1','source_host':'ssh numi','source_revision_parent':PARENT,'parent_evidence_package':'tools/evidence/connected-skin-fat-qualified-20261031b','physical_admission':False,'native_steps':0,'files':plan,'payload_file_count':len(plan),'payload_bytes':sum(e['size_bytes'] for e in plan),'checkpoint':{'path':str(cp),'sha256':sha(cp)},'qualified_proofs':{k:{'path':str(p),'sha256':sha(p)} for k,p in proofs.items()},'source_surface_crossings_reference_pose':0,'native_quality_failures':{'Skin':1111,'Fat':17640},'boundary_residuals':{'Skin_missing':0,'Skin_extra':0,'Fat_missing':135,'Fat_extra':199},'mass_mapping':'PARTIAL_CURRENT406_ONLY_NOT_REBASED_TO_CURRENT502','mass_current406_kg':mr['mass_kg'],'large_host_dependency_pending_release_archive':oversized,'nonduplicated_host_intermediate_arrays':omitted,'not_found_optional_historical_directories':missing_dirs,'historical_failed_syntax_sources':syntax,'active_future_worker_states_excluded':['fat comp0 preparation034+','nextD480batch','current406 mass exchanges026+'],'elapsed_s':time.monotonic()-start,'available_before_bytes':before}
(PKG/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(PKG/'README.md').write_text("""# Physical skin/fat construction checkpoint 502

The current O484, I local-flip and D475 plus warning-flip surfaces jointly have zero exact changed-versus-complete and zero changed-self crossing pairs against both walls and all 185 retained anatomical barriers. The combined check compares all three current surfaces against checkpoint475; individual regional passes are not substituted for this joint check. This remains a reference-pose result.

The D475 batch retains 1,846 contractions and reduces Fat native-quality failures from 20,443 to 18,012. Independent replay reconstructs all 805,358 output cells, materials and exact oriented boundary, checks all 11,533 replacement tetrahedra with exact Float32 signs, and preserves cells outside the affected stars. Its exact dyadic geometric-volume change is 4.648590561863849e-10 cubic metres; the separately retained binary64 aggregate differs by 2.16e-18.

Another 282 inferred interior-point relocations reduce Fat failures to 17,640. Every local and global volume is unchanged on the exact Float32 lattice. All 246,001 registered points, the complete actual boundary, and both protected connected-cavity footprints remain fixed. The first positional-row registration assumption failed; the corrected method uses the authenticated336 identity mask and an exact coordinate bijection. A stale duplicate Float32 coordinate view in the first intermediate was then detected and explicitly synchronized. Independent final replay verifies both coordinate views and all unchanged arrays.

The outer-skin batch reduces Skin failures from 1,135 to 1,111 with fixed point coordinates, back412, rim and registered routes. Default strict behavior has bit-identical matched-array parity. The optional construction mode admits only strict per-material bad-count descent, retaining three newly bad cells as explicitly recorded construction debt. Native thresholds are unchanged.

Three additional D triangle flips and one shared I triangle flip pass the complete source-context audit. A fourth proposed D flip is rejected for three exact intersections and is retained as failed evidence. The shared I flip has a real Skin volume transfer: two cells are retriangulated, all coordinates and unrelated cells are preserved, and all 392,522 target faces match exactly with zero missing/extra faces. The current Fat target is a closed oriented 464,834-face chain but still has 135 missing and 199 extra actual faces; its connected transfer is unfinished.

The independently replayed current406 mass ledger transfers 10.215274021429245 kg of the immutable 10.487108517986506 kg Fat source. Unallocated mass is 0.27183449655726 kg and lower-density deficit is 0.03898232855247649 kg. The original M/4, material, authenticated region, radius and density rules remain. Three inherited direct-volume roundoff overages are reported. This ledger is partial and has not been rebased to checkpoint502. The manifest identifies the one large existing LP input retained on the host pending release-asset archiving.

Physical admission is false. Accepted native steps are zero and motion has not run. There are still 1,111 Skin and 17,640 Fat native-quality failures. Original source330 masses and physiological/Brain/accepted-state owners remain unchanged. No simulated motion, native performance, physiological or clinical qualification is claimed. Controls, failed trials, exact operation maps, source hashes and independent receipts are preserved; unqualified later worker trials are excluded.
""")
receipt={'status':'PASS_CURRENT502_INCREMENT_SHA256_STAGING','package':str(PKG),'relative_package':REL,'manifest_sha256':sha(PKG/'manifest.json'),'payloads':len(plan),'payload_bytes':manifest['payload_bytes'],'physical_admission':False,'native_steps':0,'elapsed_s':time.monotonic()-start}
(OUT/'staging-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
