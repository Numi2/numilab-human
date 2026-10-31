"""Curated checkpoint publication using existing evidence-payload.v1; no native admission."""
from pathlib import Path
import json,hashlib,subprocess,os,time
H=Path(__file__).resolve().parent;P=H.parent;PUB=Path('/Users/numi/Work/human-volume-proof-publication-20261031')
PARENT='89e0665ba9e41886df98ba082d45551cea67c8b5';PKG=PUB/'tools/evidence/connected-skin-fat-qualified-20261031b';OUT=H/'publication-458-root'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def free():
 s=os.statvfs(H);return s.f_bavail*s.f_frsize
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=PUB,text=True).strip()==PARENT
assert not subprocess.check_output(['git','status','--porcelain'],cwd=PUB,text=True).strip()
proofs={
 'fixed_boundary_skin':(H/'skin-interior-LP-455-root/independent456001/receipt.json','PASS_EXACT_FIXED_BOUNDARY_INTERIOR_LP_REPLAY'),
 'D411_volume_quotient':(H/'d-shortedge-quotient-411-batch-final/volume-transfer-412/run001/independent-413-v2/independent-verification.json',None),
 'D411_comp1_volume':(H/'d-shortedge-quotient-411-batch-final/volume-transfer-414/run001/independent-415/independent-verification.json',None),
 'final_surface_context':(H/'fat-comp0-connected-warning-repair-412/run-009-final-D411-comp1-flips-exact-adapter/exact-audit/report.json',None)}
# The exact independent413 path is discovered from its pinned receipt, never guessed.
if not proofs['D411_volume_quotient'][0].exists():
 paths=list((H/'d-shortedge-quotient-411-batch-final/volume-transfer-412').rglob('*.json'))
 found=[p for p in paths if sha(p)=='5ed728315b30e1155364218f2f1fe5886c6f37017658a822ff09e14d196a9be6']
 assert len(found)==1,found;proofs['D411_volume_quotient']=(found[0],None)
for name,(p,status) in proofs.items():
 r=json.loads(p.read_text());assert r['status']==status if status else r['status'].startswith('PASS')
 assert not r.get('physical_admission',False) and r.get('native_steps',0)==0
sources=set();omitted=[];mismatch=[]
dirs=['outer-O-quotient-431-root','outer-O-quotient-433-root','outer-O-connected-sequence-440-root','outer-O-witness-driven-sequence-448-root',
 'current-skin-interior-retirement-452-root','skin-defect-domain-451-root','skin-interior-LP-454-root','skin-interior-LP-455-root',
 'fat614-comp1-target-D-refinement-407-004','d-shortedge-quotient-409-run001','d-shortedge-quotient-410-batch-final','d-shortedge-quotient-411-batch-final',
 'fat614-yy-volume-replay-004/run-037-D407-comp1-yy-prep','fat614-yy-volume-replay-004/run-039-D407-comp1-yy-prep','fat614-yy-volume-replay-004/run-040-D407-comp1-remesh','fat614-yy-volume-replay-004/run-041-D407-comp1-remesh-budget24',
 'fat-comp0-connected-warning-repair-412/run-004-d411-flips-only-control','fat-comp0-connected-warning-repair-412/run-005-d411-flips-local-context',
 'fat-comp0-connected-warning-repair-412/run-007-final-D411-comp1-flips','fat-comp0-connected-warning-repair-412/run-009-final-D411-comp1-flips-exact-adapter',
 'cold-lp-verified-COW-453-root']
# Complete current endpoints and compact exact deltas; failed/intermediate bulky
# arrays are retained on host and explicitly indexed rather than copied again.
endpoints={
 H/'outer-O-witness-driven-sequence-448-root/iteration002/typed434001/candidate-O-typed-current.npz',
 H/'skin-interior-LP-455-root/run001/conditioned-volume-proposal.npz',
 H/'current-skin-interior-retirement-452-root/run001/fixed-boundary-volume-proposal.npz',
 H/'fat614-yy-volume-replay-004/run-041-D407-comp1-remesh-budget24/connected-fixed-surface-volume-proposal.npz',
 H/'d-shortedge-quotient-411-batch-final/volume-transfer-412/run001/fat-D411-379-current406-volume-proposal.npz',
 H/'d-shortedge-quotient-411-batch-final/volume-transfer-414/run001/fat-D411-379-plus-comp1-run041-composed.npz',
 H/'fat-comp0-connected-warning-repair-412/run-007-final-D411-comp1-flips/candidate-D-three-warning-flips-typed.npz'}
for rel in dirs:
 d=H/rel
 if not d.is_dir():continue
 for p in d.rglob('*'):
  if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts or p.suffix in ('.pyc','.tmp'):continue
  if p.suffix=='.npz' and p not in endpoints and p.stat().st_size>2_000_000:
   omitted.append({'path':str(p),'sha256':sha(p),'size_bytes':p.stat().st_size,'status':'RETAINED_ON_SSH_NUMI_NOT_DUPLICATED_IN_COMPACT_PACKAGE'});continue
  if p.stat().st_size<50_000_000:sources.add(p)
for n in list(range(431,442))+[447,448,450,451,452,453,454,455,456,457]:
 for pat in ['*'+str(n)+'*root.py']:
  sources.update(H.glob(pat))
sources.update(endpoints);sources.update([H/'checkpoint.json',H/'checkpoint-fixed-skin457-root.json',Path(__file__)])
for p,_ in proofs.values():sources.add(p)
# Include actually executed producer files when their declarations match; retain
# stale historical declarations as explicit mismatches.
for p in list(sources):
 if p.suffix!='.json':continue
 try:r=json.loads(p.read_text())
 except json.JSONDecodeError:continue
 if isinstance(r,dict):
  for pathkey,hashkey in [('producer','producer_sha256'),('script','script_sha256'),('executed_source','executed_source_sha256')]:
   if isinstance(r.get(pathkey),str) and isinstance(r.get(hashkey),str):
    a=Path(r[pathkey])
    if a.is_file() and a.is_relative_to(P):
     if sha(a)==r[hashkey]:sources.add(a)
     else:mismatch.append({'receipt':str(p),'path':str(a),'declared':r[hashkey],'current':sha(a)})
plan=[];syntax=[]
for p in sorted(sources,key=str):
 assert p.is_relative_to(P) and p.is_file() and not p.is_symlink(),p
 if p.suffix=='.py':
  try:compile(p.read_text(),str(p),'exec')
  except SyntaxError as ex:syntax.append({'path':str(p),'sha256':sha(p),'error':str(ex),'status':'HISTORICAL_FAILED_SOURCE_NOT_QUALIFIED'})
 plan.append({'source_path':str(p),'payload':str(Path('payload')/p.relative_to(P)),'sha256':sha(p),'size_bytes':p.stat().st_size})
OUT.mkdir(exist_ok=False);PKG.mkdir(exist_ok=False,parents=True);start=time.monotonic();before=free()
for e in plan:
 p=PKG/e['payload'];p.parent.mkdir(exist_ok=True,parents=True);subprocess.run(['/bin/cp','-c','-p',e['source_path'],str(p)],check=True);assert sha(p)==e['sha256'] and p.stat().st_size==e['size_bytes']
manifest={'schema':'numi-human.evidence-payload.v1','source_host':'ssh numi','source_revision_parent':PARENT,'parent_evidence_package':'tools/evidence/connected-skin-fat-progress-20261031','physical_admission':False,'native_steps':0,'files':plan,'payload_file_count':len(plan),'payload_bytes':sum(e['size_bytes'] for e in plan),'qualified_proofs':{n:{'path':str(p),'sha256':sha(p)} for n,(p,_) in proofs.items()},'source_surface_crossings':0,'native_quality_failures':{'Skin':1149,'Fat_positive_comp1_D411_candidate':24174},'Fat_pre_three_flip_target_residual':{'missing':117,'extra':181},'mass_mapping':'NOT_ADMITTED_CURRENT_COMPLETE_SOURCE330_M_OVER_4_ALLOCATION_REQUIRED','nonduplicated_host_intermediate_arrays':omitted,'historical_producer_mismatches':mismatch,'historical_failed_syntax_sources':syntax,'available_before_bytes':before,'available_after_bytes':free(),'elapsed_s':time.monotonic()-start}
(PKG/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(PKG/'README.md').write_text("""# Connected tissue construction with exact replay

The final source surface D411 plus comp1 refinement and three local warning flips has zero changed-versus-complete and zero changed-self crossing pairs against current O448/I399, both walls and all 185 anatomical barriers. Its actual typed loader passes. These are reference-pose construction results, not motion qualification.

The D411 volume quotient retains 2,015 operations and removes 4,792 native-quality failures (28,942 to 24,150), with no retained-good regression. Independent replay reconstructs its exact Float32 point, cell, material and oriented boundary state. The disjoint comp1 remesh adds 24 construction-only quality failures; the composed positive Fat candidate has 819,309 cells, 253,223 points and 24,174 native-quality failures. Before the three source warning flips, its target boundary still has 117 missing and 181 extra faces. Transfer of the final flips and connected cavity remains pending.

The Skin volume improves from 1,176 to 1,149 native-quality failures: three existing interior-node retirements remove eight, and 16 new trust-region LP relocations of inferred interior points remove another 19. Controls and independent exact-integer determinant replay pass. All registered surface coordinates, material rows and complete oriented Skin boundary are preserved; the LP relocation also preserves connectivity and exact local/global volume. The deliberately inverted control initially raised before its intended rejection, and both the failed source and corrected reject are retained.

The D411 source builder avoids repeatedly converting the complete point array for local predicates. Its local-to-global exact-record control passes, and the complete 2,048-edge roster finished in 51.26 seconds. These timings describe offline construction, not native GPU simulation.

Original source330 masses are immutable. Complete conservative current-cell mass mapping is still required. No native admission thresholds, physical material equations, circulation, Brain control or accepted-state ownership change here.

Physical admission is false; native accepted steps are zero. No motion, physiological, clinical or real-time-factor qualification is claimed. The manifest lists compact omitted host intermediates and historical source mismatches explicitly. Current endpoint arrays, exact operation ledgers, tested producers, controls, failure witnesses and receipts are retained with SHA-256 identities.
""")
receipt={'status':'PASS_COMPACT_TISSUE_INCREMENT_SHA256_STAGING','package':str(PKG),'manifest_sha256':sha(PKG/'manifest.json'),'payloads':len(plan),'payload_bytes':manifest['payload_bytes'],'omitted_host_intermediates':len(omitted),'physical_admission':False,'native_steps':0,'elapsed_s':time.monotonic()-start,'available_bytes':free()}
(OUT/'staging-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
