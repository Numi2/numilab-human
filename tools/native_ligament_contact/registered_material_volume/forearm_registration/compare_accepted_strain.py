"""Compare the actual paired accepted Float32-displacement endpoints, including cells outside the fit."""
from pathlib import Path
import csv,json,sys
import numpy as np
from certify_registration_strain import compare
from compile_connected_joint import digest
def run(volume,control,trial,fit,out):
    def pose(path):
        step=list(csv.DictReader((path/'native-traces/resting-coupled.csv').open()))[-1]
        r=json.loads((path/f"native-traces/accepted-geometry/step-{step['step']}.receipt.json").read_bytes())
        if r['physical_endpoint']!='accepted' or r['accepted_step']!=int(step['step']):raise ValueError('Unverified endpoint')
        return np.loadtxt(path/'physical-accepted.node',skiprows=1,usecols=(1,2,3)),r['accepted_time_s']
    y,t=pose(control);z,tt=pose(trial)
    if t!=tt:raise ValueError('Different physical endpoints')
    x=np.loadtxt(volume/'mesh.node',skiprows=1,usecols=(1,2,3));raw=np.loadtxt(volume/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');mesh=raw[:,:4]-1
    nodes=np.asarray(json.loads((fit/'fit.json').read_bytes())['nodes_0_based']);inside=np.isin(mesh,nodes).any(1)
    moved=np.flatnonzero((y!=z).any(1));changed=np.isin(mesh,moved).any(1)
    details={};declared={}
    for region,pick in [('edited',inside),('outside',~inside)]:
        for m in (15,16):
            k=f'{region}_{m}';selected=np.flatnonzero(pick&(raw[:,4]==m)&changed);declared[k]=dict(changed=len(selected),total=int(np.sum(pick&(raw[:,4]==m))))
            failed=[];worst=None
            for i in range(0,len(selected),32768):
                sl=selected[i:i+32768];s=compare(x,y,z,mesh[sl],sl);failed.extend(s['failed'])
                if worst is None or worst['worst_ratio']<s['worst_ratio']:worst={k:v for k,v in s.items()if k not in ('failed','cells')}
            details[k]=dict(failed=failed,worst=worst)
    d=dict(time_s=t,control=digest(control/'physical-accepted.node'),trial=digest(trial/'physical-accepted.node'),source_nodes_changed=len(nodes),accepted_nodes_numerically_different=len(moved),domains=declared,results=details)
    out.write_text(json.dumps(d,indent=2)+'\n');print([(k,len(v['failed']),v['worst']['worst_ratio'] if v['worst'] else 1)for k,v in details.items()])
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
