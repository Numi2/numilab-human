"""Bounded current-Skin connected-neighborhood remesh through existing304/305."""
from pathlib import Path
import subprocess,hashlib,json,time
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
inp=H/'paired-I-interface-377-root/qualification378001/skin-target369/skin-bound-current-I-target-volume-proposal.npz'
assert sha(inp)=='0c49929963c35c904083477bcf4c41f6f79bc160b40d37e1445f2596040ddde9'
out=H/'current-skin-neighborhood-remesh-387-root';out.mkdir(exist_ok=False);started=time.monotonic();commands=[]
def call(tag,args):
 cmd=[PY,*map(str,args)];commands.append(cmd)
 with (out/(tag+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(tag,flush=True)
prep=out/'preparation304'
call('prepare304',[H/'prepare_quality_neighbor_cavities_304_root.py','--input',inp,'--output',prep,'--quality-seeds','32','--quality-rings','2','--maximum-cells','4000'])
r=json.loads((prep/'report.json').read_text())
if r['status']!='PASS_COMPACT_TARGET_CAVITY_PREPARATION':
 summary={'status':'NO_COMPLETE_PREPARED_QUALITY_NEIGHBORHOOD_WITHIN_BOUND','physical_admission':False,'native_steps':0,'preparation_report':str(prep/'report.json'),'sha256':sha(prep/'report.json'),'elapsed_s':time.monotonic()-started}
else:
 remesh=out/'remesh305'
 call('remesh305',[H/'remesh_quality_neighbor_cavities_305_root.py','--input',inp,'--output',remesh,'--prepared-dir',prep,'--prepared-sha256',sha(prep/'connected-cavity-shells.npz'),'--seconds','60','--component-seconds','1.5','--refine'])
 rr=json.loads((remesh/'report.json').read_text())
 summary={'status':'CURRENT_SKIN_CONNECTED_NEIGHBORHOOD_REMESH_FINISHED','physical_admission':False,'native_steps':0,'input':str(inp),'input_sha256':sha(inp),'preparation_report':str(prep/'report.json'),'preparation_report_sha256':sha(prep/'report.json'),'remesh_report':str(remesh/'report.json'),'remesh_report_sha256':sha(remesh/'report.json'),'passing_components':rr['passing_components'],'remaining_quality_failures':rr['remaining_quality_failures'],'candidate':rr['output'],'candidate_sha256':rr['output_sha256'],'elapsed_s':time.monotonic()-started}
summary.update(commands=commands,producer_sha256=sha(__file__))
(out/'receipt.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
