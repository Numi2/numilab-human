from pathlib import Path
import hashlib,json,subprocess,time,os
H=Path(__file__).resolve().parent;P=H.parent;PUB=Path('/Users/numi/Work/human-volume-proof-publication-20261031')
BASE='aed0a821ab90c7fc24679d2fdf8157c289b627d8'
REL='tools/evidence/connected-skin-fat-physical-transfer-20261031d'
PKG=PUB/REL;OUT=H/'publication-543-root'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=PUB,text=True).strip()
assert git('rev-parse','HEAD')==BASE and not git('status','--porcelain')
cp=H/'checkpoint-current541-root.json'
assert sha(cp)=='27f81109b00fb6b6d5c1caf128d0927ec8827e642848fb9513d00a05a2dbdeb8'
state=json.loads(cp.read_text())['current_verified_state_20261031']
assert state['surface_crossings']==0 and not state['physical_admission'] and state['native_steps']==0
assert [state[x]['native_quality_failures'] for x in ('Skin_current','Fat_current')]==[988,14366]
dirs=[
'outer-O-connected-quality-506-root','outer-O-connected-quality-530-root',
'skin-fixed-boundary-batch-508-root','skin-canonical-lineage-512-root','skin-all-views-interior-LP-515-root',
'paired-I-current-coordinate-521-controls-root','paired-I-authenticated-interior-controls-527-root',
'paired-I-current-protected-525-root','paired-I-current-protected-528-root',
'skin-canonical-paired-I-529-root','skin-canonical-outer-O-531-root','skin-canonical-fan-chain-538-root',
'fat-canonical-current-target-540-root','skin-kernel-fan-529-root','skin-kernel-fan-531-root',
'skin-node-support-cache-531-nativepreflight','skin-current-support-532','skin-current-support-538-nativepreflight',
'fat-current-support-533','global-mass-volume-preflight-532',
'lossless-frozen-native-replay-storage-534-root','lossless-older-native-evidence-storage-537-root',
'lossless-older-native-binary-storage-539-root','lossless-older-native-program-storage-542-root']
dirs+=['fat-comp0-connected-warning-repair-412/'+x for x in [
'run-034-compact-prep-481','run-035-current-target-480-comp0-interior128','run-036-reused484-volume-splice-488',
'run-037-local-conditioning-492','run-039-current-54381-normalized-rebind','run-040-current-54381-normalized-rebind',
'run-041-current-54381-normalized-rebind','run-042-current-54381-normalized-rebind',
'run-043-current-54381-500um-preflight','run-044-current-54381-500um-preflight']]
# Only completed/frozen branches, never current545 or local-cavity worker directories.
sources={cp,Path(__file__),H/'checkpoint-before-current541-root.json',H/'checkpoint-current523-root.json'}
for n in range(505,543):
 for p in H.glob('*'+str(n)+'*root.py'):
  if 'local_cavity' not in p.name:sources.add(p)
proofs={}
for key in ('Skin_current','Fat_current'):
 for key2 in ('path',):
  p=Path(state[key][key2]);assert sha(p)==state[key]['sha256'];sources.add(p)
 r=state[key]['qualification'];p=Path(r['path']);assert sha(p)==r['sha256'];proofs[key]=p
for key,entry in state['source_surfaces'].items():
 p=Path(entry['path']);assert sha(p)==entry['sha256'];sources.add(p)
for i,entry in enumerate(state['source_context_proofs']):
 p=Path(entry['path']);assert sha(p)==entry['sha256'];proofs['context_'+str(i)]=p
frozen_support=[
'fat-current-support-533/run007/independent-replay.json',
'skin-current-support-538-nativepreflight/run005/independent-replay.json']
for x in frozen_support:
 p=H/x;j=json.loads(p.read_text());assert j['status'].startswith('PASS');proofs[x]=p
# Entire completed construction directories are portable here; large redundant arrays
# are retained on the host and identified individually rather than silently discarded.
omitted=[];missing=[]
for rel in dirs:
 d=H/rel
 if not d.is_dir():missing.append(str(d));continue
 for p in d.rglob('*'):
  if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts or p.suffix in ('.pyc','.tmp'):continue
  if p.stat().st_size<=90_000_000:sources.add(p)
  else:omitted.append({'path':str(p),'sha256':sha(p),'size_bytes':p.stat().st_size,'status':'RETAINED_HOST_TOO_LARGE_ORDINARY_GIT_BLOB'})
M=P/'fractional-semantic-support-090-sparse-graph/fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001'
for p in M.iterdir():
 if p.is_dir() and any(p.name.startswith('attempt-'+str(n).zfill(3)+'-') for n in range(23,62)):
  for x in p.rglob('*'):
   if x.is_file() and not x.is_symlink() and '__pycache__' not in x.parts and x.suffix not in ('.pyc','.tmp'):
    # Keep original failed controls, executable sources, operation records and all
    # final061 arrays; do not duplicate all prior intermediate partial ledgers.
    if x.stat().st_size<2_000_000 or 'attempt-061-' in str(x.relative_to(M)):
     if x.stat().st_size<=90_000_000:sources.add(x)
     else:omitted.append({'path':str(x),'sha256':sha(x),'size_bytes':x.stat().st_size,'status':'ALREADY_RELEASE_ARCHIVED_OR_HOST_RETAINED_LARGE_INPUT'})
for x in ['frontier-diagnostic-003','source-freeze']:
 d=M/x
 if d.is_dir():
  for p in d.rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and p.stat().st_size<2_000_000:sources.add(p)
sources.update(proofs.values())
syntax=[];plan=[]
for p in sorted(sources,key=str):
 assert p.is_file() and p.is_relative_to(P) and not p.is_symlink(),p
 if p.stat().st_size>90_000_000:raise ValueError(str(p))
 if p.suffix=='.py':
  try:compile(p.read_text(),str(p),'exec')
  except SyntaxError as e:syntax.append({'path':str(p),'sha256':sha(p),'error':str(e),'status':'RETAINED_FAILED_HISTORICAL_SOURCE_NOT_QUALIFIED'})
 plan.append({'source_path':str(p),'payload':str(Path('payload')/p.relative_to(P)),'sha256':sha(p),'size_bytes':p.stat().st_size})
OUT.mkdir(exist_ok=False)
draft={'files':len(plan),'bytes':sum(x['size_bytes'] for x in plan),'omitted':len(omitted),'missing':missing}
(OUT/'plan.json').write_text(json.dumps(draft,indent=2)+'\n')
print('PLAN',json.dumps(draft),flush=True)
PKG.mkdir(parents=True,exist_ok=False);start=time.monotonic()
for e in plan:
 src=Path(e['source_path']);dst=PKG/e['payload'];dst.parent.mkdir(parents=True,exist_ok=True)
 assert sha(src)==e['sha256']
 subprocess.run(['/bin/cp','-c','-p',str(src),str(dst)],check=True)
 assert sha(dst)==e['sha256'] and dst.stat().st_size==e['size_bytes'] and sha(src)==e['sha256']
manifest={'schema':'numi-human.evidence-payload.v1','source_host':'ssh numi','source_revision_parent':BASE,
'parent_evidence_package':'tools/evidence/connected-skin-fat-physical-transfer-20261031c',
'checkpoint':{'path':str(cp),'sha256':sha(cp)},'files':plan,'payload_file_count':len(plan),
'payload_bytes':sum(x['size_bytes'] for x in plan),'qualified_proofs':{k:{'path':str(v),'sha256':sha(v)} for k,v in proofs.items()},
'physical_admission':False,'native_steps':0,'motion_verification':'NOT_RUN',
'source_surface_crossings_reference_pose':0,'native_quality_failures':{'Skin':988,'Fat':14366},
'boundary_residuals':{'Skin_missing':0,'Skin_extra':0,'Fat_missing':0,'Fat_extra':0},
'mass_mapping':'CURRENT_SUPPORT_REPLAY_PASSED_ALLOCATION_INCOMPLETE_LEGACY406_LEDGER_PARTIAL',
'legacy061_M4_metric_clarification':'Original max_M4_relative_residual label is fraction of 5e-13 tolerance; actual maximum relative closure 1.4173597439377852e-15',
'large_host_dependencies':omitted,'not_found_optional_historical_directories':missing,
'historical_failed_syntax_sources':syntax,
'active_future_worker_states_excluded':['D545 2mm quotient','paired-I local cavity fallback','current complete mass allocation'],
'elapsed_s':time.monotonic()-start}
(PKG/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(PKG/'README.md').write_text("""# Connected physical skin/fat checkpoint 541

Both volume targets are complete: Skin has 729,953 cells and 391,918 boundary faces; Fat has 798,672 cells and 459,940 boundary faces. Missing and extra target faces are zero for both layers. All cells have positive Float32 volume, but 988 Skin and 14,366 Fat cells still fail unchanged native quality criteria. These are construction candidates, not admitted physical tissue.

The O530/I528/D500 reference surfaces jointly pass the exact changed-self and changed-versus-complete audit against both layer walls and all 185 retained anatomical barriers. The back412 repair and immutable source330 mass remain unchanged. Exact cavity and quotient replay retains provenance and per-cell parent unions. Canonical current volumes synchronize both Float32 coordinate views and replace stale historical source references. The prior full parent overlap certificate is inherited only through the recorded positive, equal-boundary cavity replacements; no new whole-volume SAT or motion qualification is claimed.

The D500 batch contributes 1,277 contractions. The paired I528 batch contributes 267 contractions with per-layer volume transfer, exact zero combined geometric-volume change, and eight explicitly recorded Fat old-good-to-bad construction exchanges. The O530 batch contributes 14 contractions. Two Skin interior fan batches remove 36 failures with 30 inferred interior points, fixed original coordinates, exact boundary and zero volume change. A final no-gain fan batch is retained.

Current Skin538 and Fat540 P1 and authenticated same-material one-hop region support both pass independent replay, including full tied group sets and exact source identities. This is transfer support, not completed mass allocation. Globally, source mass fits each layer's unchanged density band (Skin 900–1100 and Fat 850–950 kg/m³), but local allocation remains incomplete.

Legacy current406 Fat ledger061 independently accounts for 10.251196682370070 kg held plus 0.23591183561643461 kg unallocated out of 10.487108517986506 kg source mass, with 0.03610959342008318 kg density-floor deficit. It has not been rebased to the current cells. A separate retained clarification resolves a misleading old metric label: 0.0028347194878755705 is a fraction of the permitted M/4 tolerance, not a relative mass error. The actual maximum relative closure is 1.4173597439377852e-15, within the existing 5e-13 gate. No mass was clipped or rescaled.

Frozen old native evidence and programs were transparently compressed with SHA-256, path and metadata preservation; no source or evidence contents were deleted. Detailed storage receipts retain measured available-space changes.

Physical admission is false, accepted native steps are zero, and motion verification has not run. Completing native cell quality and conservative local mass allocation remains necessary before physical scene integration and 16/60-second coupled motion checks. Later D545, cavity fallback and mass-allocation trials are excluded from this frozen checkpoint. Failed controls and earlier receipts remain available alongside qualified results; source geometry, transfer, physical execution, performance and physiology are separate evidence.
""")
receipt={'status':'PASS_CURRENT541_INCREMENT_SHA256_STAGING','package':str(PKG),'relative_package':REL,
'manifest_sha256':sha(PKG/'manifest.json'),'payloads':len(plan),'payload_bytes':manifest['payload_bytes'],
'physical_admission':False,'native_steps':0,'elapsed_s':time.monotonic()-start}
(OUT/'staging-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
