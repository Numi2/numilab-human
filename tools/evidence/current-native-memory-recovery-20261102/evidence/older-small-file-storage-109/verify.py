from pathlib import Path
import hashlib,json,os,time,shutil,subprocess
D=Path(__file__).resolve().parent;S=D.parent;start=time.monotonic();rows=[]
for p in [D/'replacements.jsonl',D/'remaining-002/replacements.jsonl']:
 for line in p.read_text().splitlines():
  r=json.loads(line)
  if r['phase']=='REPLACED_INDEPENDENT_COW':rows.append(r)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
failed=[];count=0
for r in rows:
 p=Path(r['target']);s=p.stat()
 if s.st_ino==Path(r['source']).stat().st_ino or sha(p)!=r['sha256'] or s.st_mtime_ns!=r['target_mtime_ns'] or s.st_mode!=r['target_mode']:failed.append(r['target'])
 count+=1
 if count%10000==0:print('verified',count,flush=True)
sourcepins=json.loads((S/'preparation-autorelease-lifetime-101/native-source-pins-104.json').read_text());root=Path('/Users/numi/Work/skin-fat-native-integration-209');sourcefail=[rel for rel,h in sourcepins.items() if sha(root/rel)!=h]
plan=json.loads((S/'native-import-106/install-motion.plan.json').read_text());inputfail=[];inputcount=0
for kind in ['input_pins','role_input_pins']:
 for raw,h in plan[kind].items():
  inputcount+=1
  if sha(raw)!=h:inputfail.append(raw)
result={'status':'PASS' if not failed and not sourcefail and not inputfail else 'FAIL','replaced_independent_paths_checked':len(rows),'new_path_failures':failed,'full_execution_source_files_checked':len(sourcepins),'execution_source_failures':sourcefail,'current_asset_and_role_pins_checked':inputcount,'current_input_failures':inputfail,'free_bytes_after':shutil.disk_usage(D).free,'swap':subprocess.run(['sysctl','vm.swapusage'],capture_output=True,text=True).stdout.strip(),'elapsed_s':time.monotonic()-start,'physical_admission':False,'accepted_full_human_steps':0};(D/'post-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);assert result['status']=='PASS'
