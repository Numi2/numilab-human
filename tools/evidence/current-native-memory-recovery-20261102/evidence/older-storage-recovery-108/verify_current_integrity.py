from pathlib import Path
import json,hashlib,time,shutil,subprocess
D=Path(__file__).resolve().parent;S=D.parent;start=time.monotonic();journals=[S/'older-small-file-storage-109/replacements.jsonl',S/'older-small-file-storage-109/remaining-002/replacements.jsonl',S/'older-compressed-frozen-storage-111/replacements.jsonl',S/'older-near-copy-storage-112/replacements.jsonl',S/'older-workspace-copy-storage-113/replacements.jsonl',S/'completed-native-copy-storage-116/replacements.jsonl'];rows={}
for p in journals:
 for line in p.read_text().splitlines():
  r=json.loads(line)
  if r['phase']=='REPLACED_INDEPENDENT_COW':rows[r['target']]=r
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
fail=[]
for i,(raw,r) in enumerate(rows.items()):
 p=Path(raw);st=p.stat()
 if sha(p)!=r['sha256'] or st.st_mode!=r['target_mode'] or st.st_mtime_ns!=r['target_mtime_ns'] or st.st_ino==Path(r['source']).stat().st_ino:fail.append(raw)
 if i and i%10000==0:print('verified',i,flush=True)
compression=[]
for p in [S/'older-large-transparent-storage-110/execution.json']:
 for r in json.loads(p.read_text())['rows']:
  if r['status']=='PASS_BYTE_IDENTICAL_TRANSPARENT_COMPRESSION':
   compression.append(r['path'])
   if sha(r['path'])!=r['sha256']:fail.append(r['path'])
sourcepins=json.loads((S/'preparation-autorelease-lifetime-101/native-source-pins-104.json').read_text());root=Path('/Users/numi/Work/skin-fat-native-integration-209');badsource=[r for r,h in sourcepins.items() if sha(root/r)!=h];plan=json.loads((S/'native-import-115/install-motion.plan.json').read_text());badinputs=[];checked=0
for domain in ['input_pins','role_input_pins']:
 for p,h in plan[domain].items():
  checked+=1
  if sha(p)!=h:badinputs.append(p)
result={'status':'PASS' if not fail and not badsource and not badinputs else 'FAIL','independent_cow_paths_checked':len(rows),'compression_paths_checked':len(compression),'storage_content_or_metadata_failures':fail,'execution_source_hashes_checked':len(sourcepins),'execution_source_failures':badsource,'current_asset_and_role_pins_checked':checked,'current_input_failures':badinputs,'free_bytes':shutil.disk_usage(D).free,'swap':subprocess.run(['sysctl','vm.swapusage'],capture_output=True,text=True).stdout.strip(),'elapsed_s':time.monotonic()-start,'accepted_full_human_steps':0,'physical_admission':False};(D/'post-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);assert result['status']=='PASS'
