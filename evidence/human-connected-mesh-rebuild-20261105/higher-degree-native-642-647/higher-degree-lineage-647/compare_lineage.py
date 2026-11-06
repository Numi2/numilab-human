from pathlib import Path
import json,hashlib,csv,importlib.util
import numpy as np
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');H=Path('/Users/numi/Work/human-checkpoint-owner-20261105');O=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('owner',H/'tools/compare_human_native_mesh_trials.py');owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
def ref(p):
 with p.open('rb') as f:return dict(path=str(p),sha256=hashlib.file_digest(f,'sha256').hexdigest())
def load(d):
 p=W/d/'matched-endpoint-channels.json';return json.loads(p.read_text()),ref(p)
old,op=load('early-state-comparison-575');parent,pp=load('stable-interior-comparison-632');new,npin=load('higher-degree-comparison-646')
assert old['candidate']['sha256']==parent['reference']['sha256']==new['reference']['sha256']
def compare(base,side,label):
 a=base[side];b=new['candidate']
 ap=Path(a['path']).parent;bp=Path(b['path']).parent
 for x,p in [(a,ap),(b,bp)]:
  ex=json.loads((p/'install-motion.execution.json').read_text());assert ex['stdout_sha256']==x['sha256'] and ex['exit_code']==0 and ex['resource_stop_reason'] is None
 diff=owner.control_differences(json.loads((ap/'install-motion.plan.json').read_text()),json.loads((bp/'install-motion.plan.json').read_text()));assert not diff,diff
 for k in ['steps','dt_s','endpoint_time_s','runtime_dt_s','runtime_coordinate_projection']:assert a['header'][k]==b['header'][k]
 def delta(x,y):return abs(y-x)/abs(x) if x else (0. if y==0 else None)
 strain={};force={};fails=[];sfails=[]
 for region in new['regional_strain']:
  x=base['regional_strain'][region][side];y=new['regional_strain'][region]['candidate']
  names=['volume_weighted_strain_norm','volume_weighted_strain_squared','maximum_green_strain_norm'];rel={k:delta(x[k],y[k]) for k in names}
  strain[region]=dict(reference=x,candidate=y,relative_change=rel)
  sfails.extend(dict(region=region,metric=k,reference=x[k],candidate=y[k],absolute_change=abs(y[k]-x[k]),relative_change=v) for k,v in rel.items() if v is None or v>.05)
 for region in new['regional_force_channels']:
  force[region]={}
  for c in new['regional_force_channels'][region]:
   x=base['regional_force_channels'][region][c][side];y=new['regional_force_channels'][region][c]['candidate'];d=float(np.linalg.norm(np.array(y['resultant_N'])-x['resultant_N']))
   values=dict(relative_resultant_difference_over_reference_nodal_l1=d/x['nodal_l1_N'] if x['nodal_l1_N'] else (0. if d==0 else None),relative_nodal_l1_change=delta(x['nodal_l1_N'],y['nodal_l1_N']))
   force[region][c]=dict(reference=x,candidate=y,resultant_difference_N=d,**values)
   fails.extend(dict(region=region,channel=c,metric=k,reference_nodal_l1_N=x['nodal_l1_N'],resultant_difference_N=d,nodal_l1_absolute_change_N=abs(y['nodal_l1_N']-x['nodal_l1_N']),relative_change=v) for k,v in values.items() if v is None or v>.01)
 trace={};ar=list(csv.DictReader((ap/'install-motion/resting-coupled.csv').open()));br=list(csv.DictReader((bp/'install-motion/resting-coupled.csv').open()));assert len(ar)==len(br)==4
 for k in ar[0]:
  x=np.array([float(r[k]) for r in ar]);y=np.array([float(r[k]) for r in br]);assert np.isfinite(x).all() and np.isfinite(y).all();d=float(np.abs(x-y).max());v=float(np.abs(x).max());trace[k]=dict(max_abs_change=d,max_abs_reference=v,relative_max_abs_change=d/v if v else (0. if not d else None))
 result=dict(reference=a,candidate=b,regional_strain=strain,regional_force_channels=force,coupled_trace=trace,source_statistics=[op,pp,npin],control_differences=diff,aggregate_reuse_only=True,physical_fidelity_qualification=False,scope='Exact retained endpoint aggregates; no new physical run or canonical pointwise qualification. Primary anatomical labels remain diagnostic.')
 (O/(label+'-channels.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 summary=dict(reference=a['path'],candidate=b['path'],force_failures=len(fails),strain_failures=len(sfails),physiology_relative_outliers=[dict(channel=k,**v) for k,v in trace.items() if v['relative_max_abs_change'] is None or v['relative_max_abs_change']>.01],force_witnesses=sorted(fails,key=lambda r:r['resultant_difference_N'],reverse=True),strain_witnesses=sorted(sfails,key=lambda r:r['absolute_change'],reverse=True),body_wall_s=dict(reference=float(a['body']['wall_s']),candidate=float(b['body']['wall_s'])),body_wall_reduction_fraction=1-float(b['body']['wall_s'])/float(a['body']['wall_s']),active_work=dict(reference=a['active_work'],candidate=b['active_work']),physical_admission=False)
 (O/(label+'-summary.json')).write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 print(label,{k:v for k,v in summary.items() if k not in ['force_witnesses','strain_witnesses','physiology_relative_outliers']},flush=True)
 return result,summary
compare(parent,'candidate','parent631');compare(old,'reference','reference495')
r,summary=compare(parent,'reference','replay572')
for key in ['regional_strain','regional_force_channels','coupled_trace']:
 assert r[key]==new[key],key
(O/'aggregate-replay-checks.json').write_text(json.dumps(dict(status='PASS_EXACT_EXISTING_COMPARATOR_METRICS',force_failures=summary['force_failures'],strain_failures=summary['strain_failures'],compared_fields=['regional_strain','regional_force_channels','coupled_trace'],sources=[op,pp,npin],analysis=ref(Path(__file__)),physical_admission=False),indent=2)+'\n')
