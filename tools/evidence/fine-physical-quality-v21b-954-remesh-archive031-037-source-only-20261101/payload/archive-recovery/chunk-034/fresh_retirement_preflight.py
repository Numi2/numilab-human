from pathlib import Path
import json,hashlib,zipfile,subprocess,datetime
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706');OUT=H/'quality-volume-archive-chunk-034-plan';plan=json.loads((OUT/'selection-plan.json').read_text());
def sha(b):return hashlib.sha256(b).hexdigest()
# Pin source receipts and rehash each selected volume file immediately before retirement.
files=[]
for row in plan['files']:
 p=Path(row['path']);s=json.loads(Path(row['source_receipt_path']).read_text());q=s[row['role']+'_current'];assert q['path']==str(p) and q['sha256']==row['expected_sha256_from_source_receipt'];assert s.get('physical_admission') is False and s.get('native_steps')==0;st=p.stat();assert p.is_file() and not p.is_symlink() and st.st_nlink==1 and st.st_size==row['size_bytes_from_stat'];dig=sha(p.read_bytes());assert dig==row['expected_sha256_from_source_receipt'];files.append({'path':str(p),'sha256':dig,'size_bytes':st.st_size,'allocated_bytes':st.st_blocks*512,'link_count':st.st_nlink,'source_receipt_path':str(Path(row['source_receipt_path'])),'source_receipt_sha256':sha(Path(row['source_receipt_path']).read_bytes()),'role':row['role'],'physical_admission':False,'native_steps':0,'native_quality_failures':q.get('native_quality_failures')})
# Check selected path/hash against all current quality-root JSON and current state-referenced NPZ metadata.
paths=[r['path'] for r in files];hashes=[r['sha256'] for r in files];roots=['all-quality-943-expanded-joint-root','all-quality-950-witness-ulp-root','all-quality-952-refinement-batch-root'];jsonpins=[];npzpaths=set();ph=[];hh=[];mh=[]
for rn in roots:
 d=H/rn
 if not d.exists():continue
 for p in d.rglob('*.json'):
  try:b=p.read_bytes();o=json.loads(b)
  except Exception:continue
  jsonpins.append({'path':str(p),'sha256':sha(b),'size_bytes':len(b)});text=b.decode('utf-8','ignore')
  for t in paths:
   if t in text:ph.append({'json':str(p),'target':t})
  for h in hashes:
   if h in text:hh.append({'json':str(p),'target_sha256':h})
  def walk(x):
   if isinstance(x,str) and x.startswith('/') and x.endswith('.npz') and Path(x).is_file():npzpaths.add(Path(x))
   elif isinstance(x,dict):
    for k,v in x.items():walk(k);walk(v)
   elif isinstance(x,list):
    for v in x:walk(v)
  walk(o)
for p in sorted(npzpaths):
 with zipfile.ZipFile(p) as z:
  for n in z.namelist():
   nm=Path(n).name.lower()
   if not n.endswith('.npy') or not any(k in nm for k in ('path','parent','sha256','source','lineage')):continue
   b=z.read(n)
   for t in paths:
    if t.encode() in b:mh.append({'npz':str(p),'member':n,'target':t})
   for h in hashes:
    if h.encode() in b:mh.append({'npz':str(p),'member':n,'target_sha256':h})
lsof=[]
for t in paths:
 r=subprocess.run(['/usr/sbin/lsof','-t',t],capture_output=True,text=True)
 if r.returncode not in (0,1):raise RuntimeError('lsof failure '+r.stderr[-300:])
 if r.stdout.strip():lsof.append({'path':t,'pids':r.stdout.strip().splitlines()})
ps=subprocess.run(['/bin/ps','-Ao','pid,command'],capture_output=True,text=True,check=True).stdout;proc=[s for s in ps.splitlines() if any(t in s for t in paths)]
# Recheck independent GH metadata and local archive bytes.
tag=plan['release_target']['tag'];api=subprocess.run(['gh','api','repos/Numi2/numilab-human/releases/tags/'+tag],capture_output=True,text=True,check=True);release=json.loads(api.stdout);a=next(x for x in release['assets'] if x['name']==plan['release_target']['asset_name']);build=json.loads((OUT/'archive-build-receipt.json').read_text());verification=json.loads((OUT/'independent-verification.json').read_text());archive=Path(plan['release_target']['local_archive_path']);assert sha(archive.read_bytes())==build['archive_sha256']==a['digest'].removeprefix('sha256:')==verification['asset_sha256'] and archive.stat().st_size==a['size'];assert verification['status'].startswith('PASS_GH_DOWNLOAD_ARCHIVE_SHA_ALL_NPZ_CRC_AND_ALL_NPY_ARRAYS_LOADED') and verification['npy_arrays_loaded']>0
assert not(ph or hh or mh or lsof or proc)
proof={'schema':'numi-human.pre824-volume-proposal-fresh-retirement-preflight.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z'),'status':'PASS_FRESH_SOURCES_GH_RESTORE_CURRENT_OWNER_AND_LIVE_HANDLE_CHECKS','chunk':'031','plan_path':str(OUT/'selection-plan.json'),'plan_sha256':sha((OUT/'selection-plan.json').read_bytes()),'archive_build_receipt_sha256':sha((OUT/'archive-build-receipt.json').read_bytes()),'independent_verification_sha256':sha((OUT/'independent-verification.json').read_bytes()),'github_release_asset':{'tag':tag,'release_id':release['id'],'asset_id':a['id'],'asset_node_id':a.get('node_id'),'asset_size':a['size'],'asset_digest':a.get('digest'),'archive_sha256':build['archive_sha256']},'selected_files':files,'current_quality_state_scan':{'roots':roots,'json_count':len(jsonpins),'json_pins':jsonpins,'state_referenced_npz_count':len(npzpaths),'npz_paths':[str(x) for x in sorted(npzpaths)],'exact_path_hits':ph,'exact_hash_hits':hh,'npz_metadata_hits':mh,'lsof_open_handles':lsof,'active_process_command_hits':proc},'preserved':{'all_source_O_I_D_arrays':True,'all_state_report_witness_files':True,'all_nonselected_files':True,'physical_admission':False,'native_steps':0},'df_free_bytes_before':__import__('shutil').disk_usage(H).free,'retirement_performed':False}
p=OUT/'fresh-retirement-preflight.json';p.write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':proof['status'],'selected_count':len(files),'logical_bytes':sum(x['size_bytes'] for x in files),'allocated_bytes':sum(x['allocated_bytes'] for x in files),'current_json_count':len(jsonpins),'current_npz_metadata_count':len(npzpaths),'path_hash_metadata_hits':0,'lsof_hits':0,'process_hits':0,'archive_sha256':build['archive_sha256'],'preflight':str(p),'preflight_sha256':sha(p.read_bytes())},sort_keys=True))
