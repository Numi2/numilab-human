"""Freeze the tested connected-layer increment in the isolated Human publication checkout."""
from pathlib import Path
import json,hashlib,subprocess,os,time,datetime
H=Path(__file__).resolve().parent;P=H.parent
PUB=Path('/Users/numi/Work/human-volume-proof-publication-20261031')
PKG=PUB/'tools/evidence/connected-skin-fat-interface-progress-20261031'
OUT=H/'publication-366-root';OUT.mkdir(exist_ok=False)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def avail():
 z=os.statvfs(H);return z.f_bavail*z.f_frsize
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=PUB,text=True).strip()=='d7145ac709a64b3ef672623bdbe356aaa360f151'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=PUB,text=True).strip()
# Only explicitly frozen stages; experimental followup364 is excluded until its own source gates pass.
directories=[
 'degree3-edge-removal-306-root/skin-batch004',
 'degree3-edge-removal-306-root/skin-batch004-independent327',
 'targeted-interior-conditioning-328-root',
 'cached-joint-witness-297-root/skin-batch002','cached-joint-witness-297-root/skin-batch003',
 'connected-skin-quality-340-root','connected-skin-quality-342-root',
 'cached-fan-continuation-344-root',
 'fat-inferred-identity-mask-336-root','fat614-yy-volume-replay-002',
 'paired-inner-interface-quotient-352-root','paired-inner-interface-quotient-353-root',
 'cached-fan-continuation-355-root','connected-skin-quality-356-root',
 'paired-I-physical-target-bind-358-root','targeted-interior-conditioning-359-root',
 'cached-joint-witness-360-root','current-skin-chain-independent-363-root',
 'current-skin-chain-independent-363-root-run002','lossless-frozen-text-storage-357-root',
 'native-boundary-edge-witness-census-350-root','reconstructed-cell-refinement-333-root/controls001',
]
sources=set()
for rel in directories:
 d=H/rel
 if not d.exists():continue
 for p in d.rglob('*'):
  if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.tmp'):sources.add(p)
for pattern in ['*327*root.py','*328*root.py','*330*root.py','*333*root.py','*336*root.py','*337*root.py','*338*root.py','*340*root.py','*342*root.py','*344*root.py','*350*root.py','*352*root.py','*353*root.py','*357*root.py','*358*root.py','*363*root.py']:
 sources.update(H.glob(pattern))
sources.add(Path(__file__));sources.add(H/'checkpoint.json')
for name in ['audit_combined_declared_surface_diff_354.py','test_declared_surface_diff_354.py']:
 sources.add(H/'combined-surface-diff-audit-270'/name)
# Include exact source/loader proof and its fully pinned dependencies.
dman=H/'fat614-minimal-f32-target-split-002/run-005/pinned-manifest.json';sources.add(dman)
dr=json.loads(dman.read_text())
for item in dr['files'].values():
 p=Path(item['path']);assert sha(p)==item['sha256'];sources.add(p)
pman=H/'fat614-minimal-f32-target-split-001/pinned-manifest.json'
pr=json.loads(pman.read_text())
for group in ['inputs','outputs']:
 for item in pr[group].values():
  if isinstance(item,dict) and 'path' in item and 'sha256' in item:
   p=Path(item['path']);assert sha(p)==item['sha256'];sources.add(p)
massman=P/'fractional-semantic-support-090-sparse-graph/fat-held-exchange-031/run-059-compact-055-056-manifest/manifest-055-056.json'
mr=json.loads(massman.read_text());sources.add(massman)
for item in mr['artifacts']:
 p=Path(item['path']);assert sha(p)==item['sha256'];sources.add(p)
# Required final proof checks before freezing any claim.
checks=[
 (H/'paired-inner-interface-quotient-353-root/complete-context-audit354001/report.json','PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'),
 (H/'cached-joint-witness-360-root/independent338001/receipt.json','PASS_INDEPENDENT_CACHED_KERNEL_FAN_CHAIN_REPLAY'),
 (H/'current-skin-chain-independent-363-root-run002/receipt.json','PASS_OMITTED_SKIN_CHAIN_INDEPENDENT_CHECKS'),
]
for p,status in checks:
 r=json.loads(p.read_text());assert r['status']==status and r['physical_admission'] is False and r['native_steps']==0
# Pin every original payload before any copy; original absolute paths remain the replay locator.
plan=[]
for p in sorted(sources,key=str):
 assert p.is_file() and not p.is_symlink()
 assert p.stat().st_size<100_000_000, ('large file must not be silently staged',p)
 try:rel=p.relative_to(P)
 except ValueError:raise ValueError(('unexpected outside source root',str(p)))
 plan.append({'source_path':str(p),'payload':str(Path('payload')/rel),'sha256':sha(p),'size_bytes':p.stat().st_size})
(OUT/'staging-plan.json').write_text(json.dumps({'schema':'numi-human.evidence-payload.v1','files':plan,'payload_bytes':sum(x['size_bytes'] for x in plan)},indent=2)+'\n')
assert avail()>12*1024**3,'restore qualification disk reserve before staging'
before=avail();PKG.mkdir(parents=True,exist_ok=False)
for item in plan:
 p=Path(item['source_path']);dest=PKG/item['payload'];dest.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['/bin/cp','-c','-p',str(p),str(dest)],check=True)
 assert dest.stat().st_size==item['size_bytes'] and sha(dest)==item['sha256']
# These are frozen evidence paths, not a new runtime format or an admission switch.
manifest={'schema':'numi-human.evidence-payload.v1','source_host':'ssh numi','source_revision_parent':'d7145ac709a64b3ef672623bdbe356aaa360f151',
 'physical_admission':False,'native_steps':0,'files':plan,'payload_file_count':len(plan),'payload_bytes':sum(x['size_bytes'] for x in plan),
 'native_quality_failures':{'Skin':1992,'Fat_after_I256_on_d4906':35806},
 'source_surface_crossings':0,'source_scope':'O4e0 / I951 / D4ce; full changed-face context, rims and all185 barriers',
 'mass_mapping':'INCOMPLETE; verified055/056 baseline142 partial ledger only; current physical geometry rebase not done',
 'fat_boundary':'INCOMPLETE','available_bytes_before_staging':before,'available_bytes_after_staging':avail()}
(PKG/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
readme="""# Connected Skin/Fat interface progress

The existing physical Skin and Fat volumes now share an audited 256-operation inner-interface quotient. It clears 135 Skin and 688 Fat native-quality failures, while their exact Float32 lattice volume changes cancel. All original coordinates remain unchanged. The full typed face unions and ancestry, 45 protected rim routes, Euler characteristic, and source identities are preserved. The emitted 814 changed interface faces have zero crossings against complete retained O/I/D surfaces, both rim-wall assets and all185 anatomical barriers, and zero joint self-intersections.

Fixed-boundary connectivity replacements, exact cavity fans, and safeguarded inferred-interior conditioning reduce the Skin endpoint from the previously published 3,302 native-quality failures to 1,992. The complete current physical boundary matches the accepted source interface. Independent owners replay the changed points, connectivity, materials, exact per-cavity volume/signs, duplicate/manifold incidence and complete oriented boundary; the omitted intermediate checks were consolidated and completed in363. No registered surface point was moved by those interior corrections.

D4ce adds eleven explicitly inferred minimally rounded Float32 boundary split points required by the existing mesher. The complete-context exact audit and actual697 metadata-view loader pass. Original source points, the protected rim and back157/158 geometry are unchanged. The Fat boundary replay remains incomplete. The I256 physical proposal has35,806 quality failures; all cells are positive. The unchanged native cell quality and volume gates still apply.

The independently replayed055 local source330 mass allocation and056 global held merge recover an additional0.106800179962kg, bringing the baseline142 Fat ledger to9.721697672190kg. Remaining source mass is0.765410845796kg, and lower-density deficit is0.064884807839kg. Per-parent M/4, positive weights, material/region and19.353mm radius controls remain enforced. This remains a partial baseline142 allocation, not a mass mapping for the changed current volumes.

Physical admission is false and this candidate has zero native accepted steps. Whole-body native integration, motion, rollback and physiology qualification remain pending. No source geometry check, loader control or positive-cell result is promoted to runtime qualification. No performance claim is made.

Failed construction and orchestration attempts are retained with the reason and correction: obsolete desired-target metadata, a missing negative-cache field, an input_npz receipt-key mismatch, unchanged TetGen refinement, the superseded mass-row indexing and serialization failures, and earlier YY replay guards. None is treated as physical evidence. Frozen original bytes, including numeric output whitespace, are preserved.

The357 storage receipt records1,228,140,544 more available bytes through transparent lossless compression and verified identical-byte clone sharing. Every original source/evidence path remains readable with the same byte hash and metadata. Actual free space is reported separately from logical allocation.

The payload uses existing NPZ/CSV/JSON evidence. Absolute source paths and hashes are replay locators; the offline tools retain their existing owner imports and require the source/evidence tree. This archive is not a standalone native scene or a new public artifact format.
"""
# Add spaces around values in prose while keeping all numerical statements unchanged.
for a,b in [('has35','has 35'),('in363','in 363'),('replayed055','replayed 055'),('and056','and 056'),('additional0.','additional 0.'),('to9.','to 9.'),('is0.','is 0.'),('and19.','and 19.'),('baseline142','baseline142'),('D4ce','D4ce'),('The357','The 357'),('records1,','records 1,')]:readme=readme.replace(a,b)
(PKG/'README.md').write_text(readme)
# Syntax check exactly the newly authored orchestration/binding source, without generating cache files.
for p in sources:
 if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
receipt={'status':'PASS_FROZEN_TESTED_EVIDENCE_STAGING','physical_admission':False,'native_steps':0,'package':str(PKG),'manifest_sha256':sha(PKG/'manifest.json'),'files':len(plan),'bytes':manifest['payload_bytes'],'available_bytes_after':avail(),'producer_sha256':sha(Path(__file__))}
(OUT/'staging-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

