
from pathlib import Path
import hashlib,json,subprocess,time,zipfile
H=Path(__file__).resolve().parent;P=H.parent;REPO='Numi2/numilab-human';TAG='human-physical-transfer502-20261031';REV='aed0a821ab90c7fc24679d2fdf8157c289b627d8'
OUT=H/'large-input-release-505-root';OUT.mkdir(exist_ok=False);start=time.monotonic()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
M=P/'fractional-semantic-support-090-sparse-graph/fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001'
src=M/'attempt-005/lp-input-current406.npz';want='4798d69dad18ffca8acedd65ec1b761135763d5121c0de67042c5cc083271328';assert sha(src)==want
manifest={'schema':'numi-human.evidence-payload.v1','source_host':'ssh numi','source_revision_parent':REV,'physical_admission':False,'native_steps':0,'scope':'single large immutable input dependency of partial current406 mass replay022; not complete allocation or native qualification','files':[{'source_path':str(src),'payload':src.name,'sha256':want,'size_bytes':src.stat().st_size}],'payload_file_count':1,'payload_bytes':src.stat().st_size}
mf=OUT/'manifest.json';mf.write_text(json.dumps(manifest,indent=2)+'\n')
notes=OUT/'release-notes.md';notes.write_text('Retains the unmodified large current406 LP input used by independently replayed partial mass ledger022. Source SHA-256: '+want+'. Original source330 M/4 is unchanged. Physical admission is false, accepted native steps are zero, and no motion qualification is claimed. Download the input and manifest; verify the complete SHA-256 before restoration. Current502 geometry and source-only qualification are committed at '+REV+'.\n')
def run(args):return subprocess.run(args,check=True,capture_output=True,text=True).stdout
probe=subprocess.run(['gh','release','view',TAG,'--repo',REPO,'--json','tagName,targetCommitish,url,assets'],capture_output=True,text=True)
if probe.returncode:
 assert 'release not found' in probe.stderr.lower(),probe.stderr
 run(['gh','release','create',TAG,'--repo',REPO,'--target',REV,'--title','Physical tissue transfer502; unadmitted','--notes-file',str(notes)])
state=json.loads(run(['gh','release','view',TAG,'--repo',REPO,'--json','tagName,targetCommitish,url,assets']))
assert state['tagName']==TAG
names={x['name'] for x in state['assets']}
for path in [src,mf]:
 if path.name not in names:run(['gh','release','upload',TAG,'--repo',REPO,str(path)])
state=json.loads(run(['gh','release','view',TAG,'--repo',REPO,'--json','tagName,targetCommitish,url,assets']))
dl=OUT/'downloaded';dl.mkdir(exist_ok=False)
for name in [src.name,mf.name]:run(['gh','release','download',TAG,'--repo',REPO,'--pattern',name,'--dir',str(dl)])
assert sha(dl/src.name)==want and (dl/src.name).stat().st_size==src.stat().st_size
assert sha(dl/mf.name)==sha(mf)
with zipfile.ZipFile(dl/src.name) as z:assert z.testzip() is None;members=z.namelist();assert members and all(x.endswith('.npy') for x in members)
assert sha(src)==want
r={'status':'PASS_GITHUB_RELEASE_DOWNLOAD_SHA256_AND_NPZ_RESTORE_CONTROL','commit':REV,'repository':REPO,'tag':TAG,'release_url':state['url'],'assets':state['assets'],'source_file':str(src),'source_sha256':want,'source_bytes':src.stat().st_size,'manifest_sha256':sha(mf),'restored_npz_array_count':len(members),'zip_crc_all_members_pass':True,'local_source_deleted':False,'current_input_preserved':True,'physical_admission':False,'native_steps':0,'producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start}
(OUT/'verified-release-receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
