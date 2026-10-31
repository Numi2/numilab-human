"""Retire only four immutable archived, superseded legacy406 partial ledgers."""
from pathlib import Path
import json,hashlib,subprocess,time,zipfile,os,shutil
from urllib.parse import quote
from urllib.request import Request,urlopen
H=Path(__file__).resolve().parent
REV='0c42755e0834759d31f76ff5b2399d30ccd8d317'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run():
 tic=time.monotonic();out=H/'verified-legacy406-ledger-retirement-601-root';out.mkdir(exist_ok=False)
 before=shutil.disk_usage(H).free;entries=json.loads((H/'verified-retirement-legacy406-census-596-root.json').read_text());assert len(entries)==4
 checkpoint=H/'checkpoint-current598-root.json';assert sha(checkpoint)=='f5be01b9695bfbb0bee0e9f63967a54c8a9641d0b425a010353e8b282356a2ac'
 best=H.parent/'fractional-semantic-support-090-sparse-graph/fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001/attempt-061-exact-volume-060-source-exhaustion/run-001/merged-held-ledger.npz';assert best.is_file()
 besthash=sha(best);rows=[]
 for i,e in enumerate(entries):
  src=Path(e['source_path']);assert src.is_file() and src!=best and src.stat().st_size==e['size_bytes'] and sha(src)==e['sha256']
  remote='tools/evidence/'+e['package']+'/'+e['payload']
  endpoint='repos/Numi2/numilab-human/contents/'+quote(remote,safe='/')+'?ref='+REV
  dest=out/('download-recovery-'+str(i)+'.npz')
  token=subprocess.check_output(['gh','auth','token'],text=True).strip()
  request=Request('https://api.github.com/'+endpoint,headers={'Accept':'application/vnd.github.raw+json','Authorization':'Bearer '+token,'User-Agent':'Numi-Human-evidence-recovery'})
  with urlopen(request,timeout=120) as response,dest.open('wb') as f:shutil.copyfileobj(response,f,length=1024*1024)
  del token,request
  assert dest.stat().st_size==e['size_bytes'] and sha(dest)==e['sha256']
  with zipfile.ZipFile(dest) as z:assert z.testzip() is None and len(z.namelist())>0;names=z.namelist()
  # Fresh source and ownership check after the independent GitHub recovery.
  opened=subprocess.run(['lsof','-t','--',str(src)],capture_output=True,text=True)
  assert opened.returncode==1 and not opened.stdout.strip(),opened.stdout
  assert src.stat().st_size==e['size_bytes'] and sha(src)==e['sha256']
  assert sha(checkpoint)=='f5be01b9695bfbb0bee0e9f63967a54c8a9641d0b425a010353e8b282356a2ac' and sha(best)==besthash
  allocated=src.stat().st_blocks*512;src.unlink();dest.unlink()
  row={**e,'immutable_commit':REV,'remote_path':remote,'downloaded_sha256_verified':True,'restoration_npz_crc_and_members_pass':True,'npz_members':names,'source_open_file_check':'NO_OPEN_HANDLES','retired_allocated_bytes':allocated,'restoration_command':['gh','api',endpoint,'-H','Accept: application/vnd.github.raw+json']}
  rows.append(row);(out/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
  print(json.dumps({'retired':str(src),'allocated':allocated,'independent_GitHub_restore':'PASS'}),flush=True)
 r={'status':'PASS_ARCHIVED_LEGACY_PARTIAL_LEDGERS_RECOVERED_BEFORE_LOCAL_RETIREMENT','entries':rows,'retained_best_legacy_061':{'path':str(best),'sha256':besthash},'available_before':before,'available_after':shutil.disk_usage(H).free,'elapsed_s':time.monotonic()-tic,'current_sources_and_checkpoint_unchanged':True,'producer_sha256':sha(Path(__file__))}
 (out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='entries'}),flush=True)
if __name__=='__main__':run()
