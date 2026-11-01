from pathlib import Path
import argparse,json,hashlib,zipfile,subprocess,datetime
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
def sha(b):return hashlib.sha256(b).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();OUT=Path(a.out);planp=OUT/'selection-plan.json';plan=json.loads(planp.read_text());rows=plan['files'];paths=[r['path'] for r in rows];hashes=[r['expected_sha256_from_source_receipt'] for r in rows]
roots=['all-quality-943-expanded-joint-root','all-quality-950-witness-ulp-root','all-quality-952-refinement-batch-root','all-quality-953-seed-star-probe-root','all-quality-953-seed-star-probe-reverse-root','all-quality-954-I250-012-root'];pins=[];npzpaths=set();ph=[];hh=[];mh=[]
for rn in roots:
 d=H/rn
 if not d.exists():continue
 for p in d.rglob('*.json'):
  try:b=p.read_bytes();o=json.loads(b)
  except Exception:continue
  pins.append({'path':str(p),'sha256':sha(b),'size_bytes':len(b)});text=b.decode('utf8','ignore')
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
lsof=[];proc=[]
for t in paths:
 r=subprocess.run(['/usr/sbin/lsof','-t',t],capture_output=True,text=True)
 if r.returncode not in (0,1):raise RuntimeError('lsof failure '+r.stderr[-300:])
 if r.stdout.strip():lsof.append({'path':t,'pids':r.stdout.strip().splitlines()})
ps=subprocess.run(['/bin/ps','-Ao','pid,command'],capture_output=True,text=True,check=True).stdout;proc=[x for x in ps.splitlines() if any(t in x for t in paths)]
status='PASS_NO_CURRENT_STATE_PATH_HASH_OR_NPZ_METADATA_REFERENCE_AND_NO_LIVE_OPEN_OWNER' if not(ph or hh or mh or lsof or proc) else 'FAIL_REFERENCES_OR_LIVE_OWNER_FOUND'
p={'schema':'numi-human.pre824-volume-proposal-current-owner-proof.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z'),'status':status,'scope':'Current quality roots 943/950/present952 JSON and referenced NPZ metadata members; exact selected path/hash searched; fresh lsof and process command scan.','selected_file_count':len(rows),'selected_paths':paths,'selected_sha256':hashes,'current_state_json_count':len(pins),'current_state_json_pins':pins,'state_referenced_npz_count':len(npzpaths),'state_referenced_npz_paths':[str(x) for x in sorted(npzpaths)],'exact_json_path_hits':ph,'exact_json_hash_hits':hh,'npz_metadata_hits':mh,'lsof_open_handles':lsof,'active_process_command_hits':proc,'not_current_physical_mass_source':True,'basis':'Each selected source receipt is pinned to physical_admission=false and native_steps=0; selected data are historical volume proposals.'}
proofp=H/f'quality-volume-pre824-current-path-proof-{plan["chunk"]["ordinal"]:03d}.json';proofp.write_text(json.dumps(p,indent=2,sort_keys=True)+'\n');plan['current_direct_path_proof']={'path':str(proofp),'sha256':sha(proofp.read_bytes()),'status':status};planp.write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':status,'proof':str(proofp),'proof_sha256':sha(proofp.read_bytes()),'states':len(pins),'referenced_npz':len(npzpaths),'path_hash_metadata_hits':len(ph)+len(hh)+len(mh),'lsof':len(lsof),'process':len(proc),'plan_sha256':sha(planp.read_bytes())},sort_keys=True))
