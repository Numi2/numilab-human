"""Compare the two seated native trials at each accepted physical time."""
import csv,json,sys
from pathlib import Path
import numpy as np

before,after,out=map(Path,sys.argv[1:])
def rows(case,name):
    with (case/'native-traces'/name).open()as f:return list(csv.DictReader(f))
a,b=rows(before,'resting-coupled.csv'),rows(after,'resting-coupled.csv')
if len(a)!=64 or len(b)!=64 or [r['time_s'] for r in a]!=[r['time_s'] for r in b]:raise ValueError('Different accepted physical times')
result={}
for field in ['lung_volume_ml','airflow_ml_s','pleural_pa','PaO2_mmhg','PaCO2_mmhg','blood_ml','blood_error_ml','lv_ml','rv_ml','aorta_mmhg','right_atrium_ml','left_atrium_ml','aortic_ejected_ml','pulmonary_ejected_ml']:
    x=np.array([float(r[field]) for r in a]);y=np.array([float(r[field]) for r in b])
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite coupled output')
    result[field]=dict(max_abs_difference=float(np.max(abs(x-y))),before_range=[float(x.min()),float(x.max())],after_range=[float(y.min()),float(y.max())])
def bed(case):
    rec=rows(case,'resting-com-support-impulses.csv');x=np.zeros((64,32,3));keys=set()
    for r in rec:
        k=int(r['accepted_step'])-1,int(r['contact_index'])
        if k in keys or int(r['segment_steps'])!=1:raise ValueError('Duplicate or mixed contact interval')
        keys.add(k);x[k]=[float(r['impulse_world_'+axis+'_ns']) for axis in 'xyz']
    if len(keys)!=2048:raise ValueError('Missing bed reactions')
    q=json.loads((case/'launch.json').read_bytes())['argv']
    return x/float(q[q.index('--muscle-step-seconds')+1])
v0=bed(before);v1=bed(after)
weights=lambda v:np.linalg.norm(v.sum(axis=1),axis=1)
forces=dict(max_total_vector_change_n=float(weights(v0-v1).max()),max_per_region_vector_change_n=float(np.linalg.norm(v0-v1,axis=2).max()),
            first_total_n=v1[0].sum(axis=0).tolist(),last_total_n=v1[-1].sum(axis=0).tolist(),
            full_interval_reaction_relative_l2=float(np.linalg.norm(v1-v0)/np.linalg.norm(v0)))
r=dict(frames=len(a),elapsed_simulated_s=float(b[-1]['time_s']),reacted_bed_loads=forces,coupled=result)
out.write_text(json.dumps(r,indent=2)+'\n')
print(forces)
