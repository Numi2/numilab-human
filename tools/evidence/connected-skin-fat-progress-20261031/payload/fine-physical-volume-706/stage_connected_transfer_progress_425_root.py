"""Archive qualified joint tissue construction progress; no native admission."""
from pathlib import Path
import json,hashlib,subprocess,os,time
H=Path(__file__).resolve().parent;P=H.parent
PUB=Path('/Users/numi/Work/human-volume-proof-publication-20261031')
PARENT='2f67b3f42adf322bf8e3a229571fd348f458018e'
PKG=PUB/'tools/evidence/connected-skin-fat-progress-20261031'
OUT=H/'publication-425-root'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def free():
 s=os.statvfs(H);return s.f_bavail*s.f_frsize
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=PUB,text=True).strip()==PARENT
assert not subprocess.check_output(['git','status','--porcelain'],cwd=PUB,text=True).strip()
proofs={
 'joint_surface_context':(H/'joint-D309-I399-context-398-root/receipt.json','4834c6a8579d5d865c032a0656862132c41b810cdfd4bfd706be122aacb24b2c'),
 'combined_fat_replay':(H/'combined-D309-I399-volume-404-root/independent-verification-002.json','661b48b3f1eac78b7ad694aca8970854d963f387689e7d2f5c8387fb4d2519ab'),
 'skin_connectivity':(H/'current-skin-fixed-boundary-421-root/receipt.json',None),
 'current_mass_partial':(P/'fractional-semantic-support-090-sparse-graph/fat-held-exchange-031/rebase-current-406/fat-run-002-current-region/independent-ledger-replay-v3.json','601ee4c418a076288cf7559f188188dc68abe82eae341f00e85034388e348fc2')}
for name,(p,pin) in proofs.items():
 if pin:assert sha(p)==pin,(name,sha(p))
 r=json.loads(p.read_text());assert r['status'].startswith('PASS_'),(name,r['status'])
 assert r.get('physical_admission',False) is False and r.get('native_steps',0)==0
sources=set()
def add_dir(d):
 assert d.is_dir(),d
 for p in d.rglob('*'):
  if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.tmp'):sources.add(p)
dirs=['paired-I-interface-392-root','paired-I-interface-394-root','paired-I-interface-399-root',
 'd-shortedge-quotient-385-run006','combined-D309-I399-volume-404-root',
 'joint-D309-I399-context-398-root','current-D309-I399-target-406-root','current-skin-fixed-boundary-421-root',
 'verified-duplicate-clones-402-root','verified-duplicate-clones-403-root',
 'connected-region-mass-support-143-current406-root']
for n in dirs:add_dir(H/n)
for pat in ['*392*root.py','*393*root.py','*394*root.py','*395*root.py','*397*root.py','*398*root.py','*399*root.py','*401*root.py','*402*root.py','*403*root.py','*404*.py','*406*root.py','*406*frozen.py','*410*.py','*421*root.py']:
 sources.update(H.glob(pat))
sources.update([H/'prepare_d_short_edge_preflight_385_run006.py',H/'checkpoint.json',H/'checkpoint-before-current-D309-I399-410-root.json',Path(__file__)])
B=P/'fractional-semantic-support-090-sparse-graph/fat-held-exchange-031'
for n in ['run-071-held-parent-expansion/attempt-011-continue-priced-closure','run-073-release-next8-from-round6-attempt014','rebase-current-406/fat-run-002-current-region','rebase-current-406/hybrid-volume-audit-001']:
 if (B/n).is_dir():add_dir(B/n)
# Pin-linked files directly referenced by current support proof, without treating unresolved historic paths as executed sources.
unavailable=[];mismatched=[]
def collect(x,source):
 if isinstance(x,dict):
  if isinstance(x.get('path'),str) and isinstance(x.get('sha256'),str) and len(x['sha256'])==64:
   p=Path(x['path'])
   if p.is_absolute() and p.is_file() and p.is_relative_to(P):
    if sha(p)==x['sha256']:sources.add(p)
    else:mismatched.append({'receipt':str(source),'path':str(p),'declared_sha256':x['sha256'],'current_sha256':sha(p)})
   elif p.is_absolute() and p.is_relative_to(P):unavailable.append({'receipt':str(source),'path':str(p),'declared_sha256':x['sha256']})
  for v in x.values():collect(v,source)
 elif isinstance(x,list):
  for v in x:collect(v,source)
for p in list(sources):
 if p.suffix=='.json':
  try:collect(json.loads(p.read_text()),p)
  except json.JSONDecodeError:pass
# Geometry/registration inputs newly produced here are included with their producers. Large immutable parent artifacts remain in the explicit parent package.
syntax_failures=[];plan=[]
for p in sorted(sources,key=str):
 assert p.is_file() and not p.is_symlink() and p.is_relative_to(P) and p.stat().st_size<100_000_000,p
 if p.suffix=='.py':
  try:compile(p.read_text(),str(p),'exec')
  except SyntaxError as e:syntax_failures.append({'path':str(p),'sha256':sha(p),'error':str(e),'status':'RETAINED_HISTORICAL_FAILED_SOURCE_NOT_EXECUTABLE_PROOF'})
 plan.append({'source_path':str(p),'payload':str(Path('payload')/p.relative_to(P)),'sha256':sha(p),'size_bytes':p.stat().st_size})
assert not syntax_failures,syntax_failures
OUT.mkdir(exist_ok=False);PKG.mkdir(parents=True,exist_ok=False)
before=free();started=time.monotonic()
for x in plan:
 dst=PKG/x['payload'];dst.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['/bin/cp','-c','-p',x['source_path'],str(dst)],check=True)
 assert sha(dst)==x['sha256'] and dst.stat().st_size==x['size_bytes']
manifest={'schema':'numi-human.evidence-payload.v1','source_host':'ssh numi','source_revision_parent':PARENT,'parent_evidence_package':'tools/evidence/current-skin-fat-physical-transfer-20261031','physical_admission':False,'native_steps':0,'files':plan,'payload_file_count':len(plan),'payload_bytes':sum(x['size_bytes'] for x in plan),'qualified_proofs':{k:{'path':str(p),'sha256':sha(p)} for k,(p,_) in proofs.items()},'source_surface_crossings':0,'native_quality_failures':{'Skin':1506,'Fat':28942},'Fat_boundary_residual':{'missing':127,'extra':197},'mass_mapping':'PARTIAL_CURRENT406_RETAINED_LEDGER_NOT_ADMITTED','unavailable_historical_receipt_paths':unavailable,'historic_source_or_artifact_hash_mismatches':mismatched,'available_before_bytes':before,'available_after_bytes':free(),'elapsed_s':time.monotonic()-started}
(PKG/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(PKG/'README.md').write_text("""# Connected physical skin/fat construction progress

The current joint source surfaces have zero crossings at the audited reference pose. Outer skin and the back repair are unchanged. The D309/I399 combined increment passed exact regional cell replay and a joint audit of 9,594 changed faces against both adjacent layers, walls, and all 185 registered anatomical barriers.

The fully bound current Fat volume has 826,737 positive cells and 28,942 cells below the existing native quality threshold. Its boundary still has 127 missing and 197 extra target faces. Skin has a complete 393,364-face target boundary and 1,506 remaining quality failures; six existing fixed-boundary connectivity operations removed seven failures without moving coordinates or changing material volumes.

The current406 semantic support and retained mass-ledger subset pass independent replay. The subset carries 9.3078501884 kg; 1.1792583296 kg of source Fat mass remains unallocated. The low-density deficit is 0.5353714283 kg and the maximum source-corner M/4 residual remains 1.0. This is not complete conservative physical admission. Numeric volume and density roundoff is recorded in the detailed reports.

This package preserves construction code, exact source hashes, candidates, controls and failed trials. It extends the named parent evidence package; original source assets and the full native runtime are not replicated here. Historical receipt paths with missing or changed working bytes are enumerated explicitly instead of being represented as recovered executed source.

Physical admission is false. Native accepted steps are zero. No motion, physiological, clinical or simulation throughput qualification is claimed. Offline geometry construction timings do not measure native real-time factor.
""")
(OUT/'staging-receipt.json').write_text(json.dumps({'status':'PASS_SHA256_SIZE_VERIFIED_COW_STAGING','package':str(PKG),'manifest_sha256':sha(PKG/'manifest.json'),'payloads':len(plan),'payload_bytes':manifest['payload_bytes'],'physical_admission':False,'native_steps':0,'available_after_bytes':free(),'elapsed_s':time.monotonic()-started},indent=2)+'\n')
print(json.dumps({'package':str(PKG),'payloads':len(plan),'payload_bytes':manifest['payload_bytes'],'manifest_sha256':sha(PKG/'manifest.json'),'unavailable_history_count':len(unavailable),'mismatched_history_count':len(mismatched),'available_bytes':free()}),flush=True)
