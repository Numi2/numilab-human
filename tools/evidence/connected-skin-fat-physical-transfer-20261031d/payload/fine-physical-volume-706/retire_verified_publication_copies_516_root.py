
"""Retire only independently restored publication copies and task-owned download cache."""
from pathlib import Path
import subprocess,json,hashlib,shutil,time,os
H=Path(__file__).resolve().parent
PUB=Path('/Users/numi/Work/human-volume-proof-publication-20261031')
OUT=H/'publication-copy-retirement-516-root';OUT.mkdir(exist_ok=False)
ROOT=H/'remote-publication-restore-509-root';receipt=ROOT/'receipt.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(a):return subprocess.run(a,check=True,capture_output=True,text=True).stdout
r=json.loads(receipt.read_text());assert r['status']=='PASS_INDEPENDENT_GITHUB_FETCH_FULL_BYTE_RESTORE_AND_NPZ_CRC'
mp=Path(r['manifest']);assert sha(mp)==r['manifest_sha256']
m=json.loads(mp.read_text());rev=r['revision'];assert rev=='aed0a821ab90c7fc24679d2fdf8157c289b627d8'
assert run(['git','-C',str(PUB),'rev-parse','HEAD']).strip()==rev
assert not run(['git','-C',str(PUB),'status','--porcelain']).strip()
assert run(['git','-C',str(PUB),'sparse-checkout','list']).strip()=='/tools/evidence/fine-volume-206-209-205-20261031/**'
remote=run(['git','-C',str(PUB),'ls-remote','origin','refs/heads/main']).split()[0];assert remote==rev
ps=run(['ps','-Ao','pid,ppid,etime,command'])
assert not any(str(PUB/'tools/evidence') in l or str(ROOT/'independent.git') in l for l in ps.splitlines())
(OUT/'process-ownership-before.txt').write_text(ps)
before=shutil.disk_usage(H).free;start=time.monotonic()
for e in m['files']:
 p=PUB/e['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==e['bytes'] and sha(p)==e['sha256']
run(['git','-C',str(PUB),'sparse-checkout','reapply'])
assert not run(['git','-C',str(PUB),'status','--porcelain']).strip()
missing=[e for e in m['files'] if not (PUB/e['path']).exists()]
assert len(missing)>10000
cache=ROOT/'independent.git';assert cache.resolve()==cache and cache.parent==ROOT and (cache/'shallow').is_file()
shutil.rmtree(cache)
# The redundant release download was also independently verified, but its source stays intact.
large=H/'large-input-release-505-root'
vr=large/'verified-release-receipt.json'
download_deleted=False
if vr.exists():
 v=json.loads(vr.read_text());assert v['status']=='PASS_GITHUB_RELEASE_DOWNLOAD_SHA256_AND_NPZ_RESTORE_CONTROL'
 dl=large/'downloaded';assert dl.resolve()==dl and dl.parent==large
 assert sha(Path(v['source_file']))==v['source_sha256']
 restored=dl/'lp-input-current406.npz'
 assert sha(restored)==v['source_sha256']
 shutil.rmtree(dl);download_deleted=True
after=shutil.disk_usage(H).free
res={'status':'PASS_VERIFIED_REVERSIBLE_PUBLICATION_COPY_RETIREMENT','repository':'Numi2/numilab-human','revision':rev,'independent_restore_receipt':str(receipt),'independent_restore_receipt_sha256':sha(receipt),'restore_manifest_sha256':sha(mp),'removed_redundant_worktree_files':len(missing),'removed_logical_worktree_bytes':sum(e['bytes'] for e in missing),'independent_private_git_download_cache_removed':True,'verified_release_download_copy_removed':download_deleted,'current_source_and_physics_evidence_unchanged':True,'publication_git_status_clean':True,'restore_command':'git -C '+str(PUB)+' restore --source='+rev+' --worktree --ignore-skip-worktree-bits -- tools/evidence','available_bytes_before':before,'available_bytes_after':after,'observed_available_gain_bytes':after-before,'qualification_floor_bytes':12*1024**3,'qualification_floor_met':after>=12*1024**3,'physical_admission':False,'native_steps':0,'producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start}
(OUT/'receipt.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res),flush=True)
