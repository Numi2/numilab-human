"""Freeze the tested current physical transfer and indexed construction owners."""
from pathlib import Path
import json,hashlib,subprocess,os
H=Path(__file__).resolve().parent;P=H.parent
PUB=Path('/Users/numi/Work/human-volume-proof-publication-20261031')
PARENT='26b610492a75f920570cd4fc33574dc1bd923c5f'
PKG=PUB/'tools/evidence/current-skin-fat-physical-transfer-20261031';OUT=H/'publication-390-root'
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def free():
 s=os.statvfs(H);return s.f_bavail*s.f_frsize
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=PUB,text=True).strip()==PARENT
assert not subprocess.check_output(['git','status','--porcelain'],cwd=PUB,text=True).strip()
proof_names=[
 'paired-I-interface-364-root/typed-replay365001/receipt.json',
 'paired-I-interface-364-root/complete-context-audit354001/report.json',
 'paired-D307-I364-volume-composition-007/independent-replay-004/report.json',
 'current-skin-quality-373-root/receipt.json',
 'reconstructed-cell-refinement-375-root/independent376001/receipt.json',
 'paired-I-interface-377-root/qualification378001/receipt.json',
 'current-skin-neighborhood-remesh-387-root/independent388001/receipt.json',
 'indexed-interface-owner-381-root/receipt.json','matched-indexed-interface-386-root/receipt.json']
for rel in proof_names:
 r=json.loads((H/rel).read_text());assert r['status'].startswith('PASS_') and r['physical_admission'] is False and r['native_steps']==0
dirs=[
 'd-shortedge-quotient-351-run002','paired-I-interface-364-root','paired-I-physical-target-369-root','connected-skin-quality-370-root',
 'paired-D307-I364-volume-composition-007','paired-D307-I364-full-domain-008',
 'current-skin-quality-373-root','reconstructed-cell-refinement-375-root','paired-I-interface-377-root',
 'indexed-interface-owner-381-root','old-publication-pack-readonly-inventory-382-root',
 'inactive-publication-pack-dedup-383-root','matched-indexed-interface-386-root','current-skin-neighborhood-remesh-387-root']
for rel in ['run-004-interior-node3386-v4','run-006-interior-155-55-sat','run-007-comp7-current-plc','run-009-bounded-stop','run-010-composed-boundary-components-v4','run-011-rebased-377-007-components','run-014-current-target-rebound-v3']:
 dirs.append('fat614-yy-volume-replay-004/'+rel)
sources=set()
def add_dir(d):
 assert d.is_dir(),d
 for p in d.rglob('*'):
  if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.tmp'):sources.add(p)
for rel in dirs:add_dir(H/rel)
for n in ['364','365','369','373','374','376','378','379','380','381','383','384','386','387','388']:sources.update(H.glob('*'+n+'*root.py'))
sources.update([Path(__file__),H/'checkpoint.json'])
fat=H/'fat614-yy-volume-replay-004'
for name in ['replay_remaining_audited_yy_components_v1.py','replay_remaining_audited_yy_components_v2.py','replay_remaining_audited_yy_components_v3_corrected.py','replay_remaining_audited_yy_components_v4.py','rebase_yy_components_onto_5bf377_v1.py','rebind_fat_target_from_typed_surfaces_v1.py','rebind_fat_target_from_typed_surfaces_v2.py','rebind_fat_target_from_typed_surfaces_v3.py']:sources.add(fat/name)
dman=H/'d-shortedge-quotient-351-run002/publication-manifest-001.json'
for x in json.loads(dman.read_text())['files']:
 p=Path(x['path']);assert sha(p)==x['sha256'];sources.add(p)
mass=P/'fractional-semantic-support-090-sparse-graph/fat-held-exchange-031'
mm=mass/'run-061-global-held-merge-top13-060/attempt-004/independent-verification-manifest.json';mr=json.loads(mm.read_text())
assert mr['status']=='PASS_SOURCE_ONLY_060_PROPOSAL_AND_061_MERGE_REPLAY';sources.add(mm)
for x in mr['inputs'].values():
 p=mass/x['path'];assert sha(p)==x['sha256'];sources.add(p)
iv=mr['independent_verifier']
for key,hkey in [('path','sha256'),('receipt','receipt_sha256')]:
 p=mass/iv[key];assert sha(p)==iv[hkey];sources.add(p)
for rel in ['run-060-top13-exact-tet-closure/attempt-001','run-060-top13-exact-tet-closure/attempt-002','run-061-global-held-merge-top13-060/attempt-004']:
 if (mass/rel).exists():add_dir(mass/rel)
def collect(x):
 if isinstance(x,dict):
  if isinstance(x.get('path'),str) and isinstance(x.get('sha256'),str):
   p=Path(x['path'])
   if p.is_file():assert sha(p)==x['sha256'];sources.add(p)
  for v in x.values():collect(v)
 elif isinstance(x,list):
  for v in x:collect(v)
collect(json.loads((fat/'run-014-current-target-rebound-v3/report.json').read_text()))
plan=[];retained_failed_syntax=[]
for p in sorted(sources,key=str):
 assert p.is_file() and not p.is_symlink() and p.stat().st_size<100_000_000,p
 if p.suffix=='.py':
  try:compile(p.read_text(),str(p),'exec')
  except SyntaxError as e:
   allowed=H/'paired-D307-I364-volume-composition-007/independent-check-source-001/independent_check_composition_007.corrected-002.py'
   assert p==allowed, ('unexpected current-source syntax failure',str(p),str(e))
   retained_failed_syntax.append({'source_path':str(p),'sha256':sha(p),'status':'ARCHIVED_FAILED_CHECKER_PREPARATION_NOT_EXECUTABLE_PROOF','line':e.lineno,'error':str(e)})
 plan.append({'source_path':str(p),'payload':str(Path('payload')/p.relative_to(P)),'sha256':sha(p),'size_bytes':p.stat().st_size})
assert free()>=12*1024**3
OUT.mkdir(exist_ok=False);before=free()
(OUT/'staging-plan.json').write_text(json.dumps({'schema':'numi-human.evidence-payload.v1','files':plan},indent=2)+'\n')
PKG.mkdir(parents=True,exist_ok=False)
for x in plan:
 dst=PKG/x['payload'];dst.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['/bin/cp','-c','-p',x['source_path'],str(dst)],check=True)
 assert sha(dst)==x['sha256'] and dst.stat().st_size==x['size_bytes']
manifest={'schema':'numi-human.evidence-payload.v1','source_host':'ssh numi','source_revision_parent':PARENT,'physical_admission':False,'native_steps':0,'files':plan,'retained_failed_source_syntax':retained_failed_syntax,'payload_file_count':len(plan),'payload_bytes':sum(x['size_bytes'] for x in plan),'native_quality_failures':{'Skin':1675,'Fat':32526},'Fat_target_boundary':{'faces':483534,'missing':127,'extra':197,'complete':False},'source_surface_crossings':0,'mass_mapping':'Incomplete baseline142 ledger; current-mesh final mappings pending','available_before_bytes':before,'available_after_bytes':free()}
(PKG/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
readme="""# Current Skin/Fat physical-transfer increment

Paired inner-interface batches of 1,284 and 941 operations repair Skin and Fat together with unchanged Float32 point coordinates and exactly cancelling combined volume changes. Full typed ancestry, rim, independent volume and complete-context intersection checks pass. The latest 3,209 changed source faces have zero crossings against retained surfaces, both walls and all 185 anatomical barriers. Outer skin and back geometry are unchanged.

Skin now has 1,675 native-quality failures, down from 1,992 in the parent publication. Its complete 398,702-face target boundary matches; all cells are positive. Independent replay covers point, material, connectivity, exact local sign/volume and complete oriented boundary. The small neighborhood remesh and its failed alternatives are retained.

D307 and the combined D/I volume replay pass. Seven Fat boundary components are rebased onto the current interface: all cells positive, 4,065 internal cuts matching bidirectionally, exact retained/interpatch SAT passing. Fat has 32,526 native-quality failures. The complete typed target has 483,534 faces; 127 missing and 197 extra actual-boundary faces remain. Target rebind corrects stale I metadata and maps wall-node IDs through the collar namespace. Its geometry arrays are unchanged.

Cached incidence replaces per-trial whole-body scans with CSR lookup and explicit retained-endpoint updates. Actual-owner controls match 64 operations per tissue and every changed keep star. A complete matched 941-operation batch gives byte-identical source, Skin and Fat NPZs: 67.3853 seconds before and 14.5584 after, a 4.63x improvement in offline repair throughput. This is not simulation real-time performance.

Independent060/061 replay confirms the incomplete baseline142 Fat ledger: 9.78964022308 kg held, 0.697468294907 kg source mass remaining and 0.0197570439013 kg below the density minimum. It checks M/4, material/region, hard radius and whole-ledger accounting. This is not the final mass mapping for these changed volumes; Skin also lacks a complete final mapping.

Storage383 recovered 913,076,224 available bytes. All 11,204 retired-pack objects remain available; 10,187 shared objects were content-hash verified and 1,017 unique objects retained in a verified replacement pack. Every Git ref, source and evidence payload is preserved.

Resume from current_verified_state_20261031 in the frozen checkpoint. Older fields remain historical evidence. The archived corrected-002 checker preparation contains a syntax error and is retained as a failed attempt; the qualified independent checker is corrected-004. Physical admission is false, accepted native steps are zero, and native motion/rollback/physiology qualification remains pending. The body is a mixed-source reference assembly, not one measured person. This uses existing artifact formats and source replay locators; it requires the owning source/evidence tree.
"""
(PKG/'README.md').write_text(readme)
r={'status':'PASS_TESTED_CURRENT_TISSUE_INCREMENT_STAGING','physical_admission':False,'native_steps':0,'package':str(PKG),'manifest_sha256':sha(PKG/'manifest.json'),'payload_files':len(plan),'payload_bytes':manifest['payload_bytes'],'available_after_bytes':free(),'producer_sha256':sha(__file__)}
(OUT/'staging-receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
