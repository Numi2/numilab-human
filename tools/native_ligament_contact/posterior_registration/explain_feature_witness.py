"""Identify which current contact rows change the actual nodal reaction during a strict native trial."""
import csv,json,sys
from pathlib import Path
import numpy as np
from material_context import sha
def summarize(assets,split,features,out):
    pool=np.fromfile(assets/'pool-material-points.bin','f4').reshape(-1,7,4);reverse=json.loads((assets/'manifest.json').read_text())['closed_reverse']['first']
    with split.open() as f:parts=list(csv.DictReader(f))
    with features.open() as f:sites=list(csv.DictReader(f))
    def vector(s):return np.fromstring(s,sep=' ')
    result={}
    for stage in (0,1,15):
        for side in (0,1):
            row=next(r for r in parts if int(r['endpoint'])==0 and int(r['iteration'])==stage and int(r['side'])==side);node=int(row['worst_free_node'])
            ranks=[];change=np.zeros(3)
            for z in sites:
                if int(z['iteration'])!=stage or int(z['slot'])//256!=side:continue
                oldnode=vector(z['old_nodes']).astype(int);newnode=vector(z['new_nodes']).astype(int);i=int(z['site'])
                of=vector(z['old_force_N']);nf=vector(z['new_force_N']);n1=np.linalg.norm(of);n2=np.linalg.norm(nf)
                weight1=sum(vector(z['old_weights'])[oldnode==node]);weight2=sum(vector(z['new_weights'])[newnode==node]);impact=weight2*nf-weight1*of
                if max(abs(impact))==0:continue
                change+=impact
                f=pool[i,3].copy().view('u4')
                ranks.append(dict(site=i,native_slot=int(z['slot']),type='bone vertex on material' if i>=reverse else 'material point on bone',cortical_body=int(f[2]),absolute_load_N=float(z['pressure_N']),signed_gap_m=float(z['signed_gap_m']),reaction_angle_degrees=float(np.rad2deg(np.arccos(np.clip(of@nf/(n1*n2),-1,1)))),prior_material_weight=weight1,current_material_weight=weight2,ancestry_changed=not np.array_equal(oldnode,newnode),contribution_change_N=float(np.linalg.norm(impact))))
            ranks.sort(key=lambda z:-z['contribution_change_N'])
            expected=float(row['same_node_feature_change_N']);measured=float(np.linalg.norm(change))
            if abs(measured-expected)>5e-9+5e-5*expected:raise ValueError(('Mismatched force attribution',stage,side,measured,expected))
            result[f'{side}/{stage}']=dict(worst_node=node,feature_force_change_N=measured,rows=ranks[:6])
    out.write_text(json.dumps(dict(trial_source=sha(features),nodal_probe=sha(split),witnesses=result),indent=2)+'\n')
if __name__=='__main__':summarize(*map(Path,sys.argv[1:]))
