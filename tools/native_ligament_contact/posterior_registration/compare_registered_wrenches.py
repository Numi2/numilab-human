from pathlib import Path
import csv,json,sys
import numpy as np
from material_context import sha
def reconstruct(a,z):
    d=json.loads((a/'manifest.json').read_text());k0=d['finite_facets']['start']+16*d['motion_contact_registry']['original_pairs']
    static=np.fromfile(a/'pool-material-points.bin','f4').reshape(-1,7,4)
    owners=np.fromfile(a/'motion-cortical-owners.bin','u4').reshape(-1,4)
    reg={int(r['slot']):int(r['cortical_row']) for r in csv.DictReader((z/'accepted-cortical-registry.csv').open())}
    src=z/'accepted-pelvic-state.csv';rows=list(csv.DictReader(src.open()))
    def field(name):return {int(r['index']):np.array([r[j] for j in 'xyzw'],'f4').astype(float) for r in rows if r['field']==name}
    loc=field('material_node');x=np.stack([loc[i][:3] for i in range(len(loc))])
    weights=field('reverse_shape');parents=field('reverse_parent_nodes')
    totals={};nodal=np.zeros_like(x);step=int(rows[0]['accepted_step'])
    for k,v in field('normal_and_force').items():
        if v[3]<=0:continue
        if k<k0:
            domain=static[k,3].view('u4');side=int(domain[1]==145);body=int(domain[2])
        else:
            raw,body,side,donor=owners[reg[(k-k0)//16]]
        w=weights[k] if k in weights else static[k,6]
        n=parents[k].astype(int) if k in parents else static[k,5].view('u4')
        force=v[:3]*v[3];np.add.at(nodal,n,w[:,None]*force)
        # Force on bone is the negative tissue force; both moment and load
        # are expressed in the same registered pelvic frame at this endpoint.
        totals.setdefault(f'{int(side)}/{int(body)}',np.zeros(7))[:] += np.r_[v[3],-force,np.cross(w@x[n],-force)]
    return step,sha(src),totals,nodal
a,b,x,y,out=map(Path,sys.argv[1:]);n,h,old,u=reconstruct(a,x);m,i,new,v=reconstruct(b,y)
if n!=m:raise ValueError('Compare identical accepted physical times')
report={'root':n,'control_capture':h,'trial_capture':i,'per_bone':{}}
for key in sorted(old.keys()|new.keys()):
    p=old.get(key,np.zeros(7));q=new.get(key,np.zeros(7));report['per_bone'][key]={
        'control_compressive_magnitude_sum_N':p[0],'trial_compressive_magnitude_sum_N':q[0],
        'control_force_N':p[1:4].tolist(),'trial_force_N':q[1:4].tolist(),
        'force_difference_N':np.linalg.norm(p[1:4]-q[1:4]),'control_force_norm_N':np.linalg.norm(p[1:4]),
        'moment_difference_Nm':np.linalg.norm(p[4:]-q[4:]),'control_moment_norm_Nm':np.linalg.norm(p[4:])}
report['relative_nodal_contact_force_l2']=np.linalg.norm(u-v)/np.linalg.norm(u)
report['max_nodal_force_difference_N']=np.linalg.norm(u-v,axis=1).max()
out.write_text(json.dumps(report,indent=2)+'\n')
