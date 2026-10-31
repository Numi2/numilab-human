"""Bounded outer-Skin continuation with exact-witness pruning and remembered failed edges."""
from pathlib import Path
import argparse,json,subprocess,hashlib,time
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--initial-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--cycles',type=int,default=2);args=ap.parse_args()
assert 1<=args.cycles<=4
currentroot=args.initial_root;initialproof=currentroot/'qualification435001/receipt.json'
proof=json.loads(initialproof.read_text());assert proof['status']=='PASS_BOUNDED_OUTER_SKIN_TYPED_GEOMETRY_VOLUME_CONTEXT_AND_TARGET'
tr=json.loads((currentroot/'typed434001/receipt.json').read_text())
skin=Path(proof['candidate']);surface=Path(tr['candidate']);bad=proof['Skin_native_quality_after'];before=bad
assert sha(skin)==proof['candidate_sha256'] and sha(surface)==tr['candidate_sha256']
exclusions={}
def add_exclusions(root):
 rp=root/'run001/report.json';r=json.loads(rp.read_text())
 for item in r.get('excluded_witness_reports',[]):exclusions[item['path']]=item['sha256']
 if r.get('witness_pruning'):exclusions[str(rp)]=sha(rp)
add_exclusions(currentroot)
out=args.output;out.mkdir(exist_ok=False);start=time.monotonic();history=[];absolute_delta=0.
for cycle in range(args.cycles):
 root=out/('iteration%03d'%(cycle+1));root.mkdir(exist_ok=False)
 commands=[];failed=[]
 def call(tag,argv):
  cmd=[PY,*map(str,argv)];commands.append(cmd)
  with (root/(tag+'.log')).open('w') as f:res=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  return res.returncode
 argv=[H/'repair_bounded_outer_O_with_witnesses_447_root.py','--skin',skin,'--skin-sha256',sha(skin),'--o',surface,'--o-sha256',sha(surface),'--output',root/'run001','--maximum-edge-m','.001','--maximum-operations','1024','--seconds','40']
 for path,pin in exclusions.items():argv+=['--exclude-witness-report',path,pin]
 assert call('construct447',argv)==0
 rr=json.loads((root/'run001/report.json').read_text())
 if rr['passing_operations']==0:
  history.append({'iteration':cycle+1,'status':'NO_PROGRESS_STOP','raw_report':str(root/'run001/report.json'),'sha256':sha(root/'run001/report.json')});break
 assert call('typed434',[H/'bind_and_replay_bounded_outer_O_434_root.py','--report',root/'run001/report.json','--output',root/'typed434001'])==0
 code=call('qualify439',[H/'qualify_bounded_outer_skin_generic_439_root.py','--root',root]);accepted=root
 for prune in range(4):
  if code==0:break
  apath=accepted/'qualification435001/audit354/report.json'
  a=json.loads(apath.read_text());assert a['status']=='FAIL_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
  failed.append({'root':str(accepted),'context':str(apath),'sha256':sha(apath)})
  nextroot=out/('iteration%03d-pruned%03d'%(cycle+1,prune+1))
  code=call('prune%03d'%prune,[H/'prune_witnessed_outer_skin_441_root.py','--failed-root',accepted,'--output',nextroot])
  accepted=nextroot
 assert code==0,('bounded witness repair exhausted',str(accepted))
 qp=accepted/'qualification435001/receipt.json';qr=json.loads(qp.read_text())
 assert qr['status']=='PASS_BOUNDED_OUTER_SKIN_TYPED_GEOMETRY_VOLUME_CONTEXT_AND_TARGET'
 nr=json.loads((accepted/'typed434001/receipt.json').read_text());ar=json.loads((accepted/'run001/report.json').read_text())
 skin=Path(qr['candidate']);surface=Path(nr['candidate']);newbad=qr['Skin_native_quality_after'];gain=bad-newbad;assert gain>0
 add_exclusions(accepted);absolute_delta+=ar['accumulated_absolute_volume_change_m3'];currentroot=accepted
 history.append({'iteration':cycle+1,'status':'PASS_TYPED_FULL_CONTEXT_AND_TARGET','accepted_root':str(accepted),'qualification':str(qp),'qualification_sha256':sha(qp),'bad_before':bad,'bad_after':newbad,'operations':ar['passing_operations'],'failed_contexts':failed,'commands':commands})
 bad=newbad
 progress={'status':'BOUNDED_WITNESS_DRIVEN_OUTER_SKIN_IN_PROGRESS','physical_admission':False,'native_steps':0,'history':history,'latest_qualified_root':str(currentroot),'latest_skin':str(skin),'latest_skin_sha256':sha(skin),'latest_O':str(surface),'latest_O_sha256':sha(surface),'native_quality_failures':bad}
 (out/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');print(json.dumps({k:v for k,v in history[-1].items() if k!='commands'}),flush=True)
 if gain<12:break
receipt={'status':'PASS_BOUNDED_WITNESS_DRIVEN_OUTER_SKIN_SEQUENCE','physical_admission':False,'native_steps':0,'initial_qualification':str(initialproof),'initial_qualification_sha256':sha(initialproof),'quality_before':before,'quality_after':bad,'history':history,'final_qualified_root':str(currentroot),'final_skin':str(skin),'final_skin_sha256':sha(skin),'final_O':str(surface),'final_O_sha256':sha(surface),'absolute_geometric_volume_change_this_sequence_m3':absolute_delta,'excluded_failed_edge_report_pins':exclusions,'back412_unchanged':True,'mass_mapping':'NOT_ADMITTED_CURRENT_SOURCE330_CONSERVATIVE_REBIND_REQUIRED','native_motion':'NOT_RUN','producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
