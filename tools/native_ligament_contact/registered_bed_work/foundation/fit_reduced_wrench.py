"""Find a single rigid initial pose whose distributed support balances the opening wrench."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def fit(opening,foundation):
    g=np.genfromtxt(opening/'registered-bed-columns.csv',names=True,delimiter=',');g=g[g['area_m2']>0]
    row=next(csv.DictReader((opening/'resting-surface-audit.csv').open()))
    com=np.array([float(row['body_com_'+i+'_m']) for i in 'xyz']);w=float(row['represented_body_mass_kg'])*9.81
    rel=np.stack([g[k+'_m'] for k in 'xyz'],axis=1)-com
    import re
    table=np.array([[float.fromhex(s[:-1]) for s in m.split(',')] for m in re.findall(r'float4\(([^)]+)\)',(foundation/'NumiHumanFoamFoundation.metalinc').read_text())])
    def depth(idx):return g['skin_m']*table[idx,0]+g['fat_m']*table[idx,1]+table[idx,2]
    def wrench(q):
        R=Rotation.from_rotvec([*q[1:],0.]);r=R.apply(rel);y=np.maximum(q[0]-com[2]-r[:,2],0.)
        lo=np.zeros(len(g),int);hi=lo+256
        for _ in range(8):
            mid=(lo+hi)//2;sel=depth(mid)<y;lo=np.where(sel,mid,lo);hi=np.where(sel,hi,mid)
        p=table[lo,3]+(table[hi,3]-table[lo,3])*(y-depth(lo))/(depth(hi)-depth(lo))
        f=g['area_m2']*p
        return np.array([f.sum()-w,f@r[:,1],-f@r[:,0]]),p
    res=least_squares(lambda q:wrench(q)[0],[.12,0,0],bounds=([.03,-.12,-.12],[.2,.12,.12]),ftol=1e-12,xtol=1e-12,gtol=1e-12)
    v,p=wrench(res.x)
    if abs(v).max()>1e-5 or p.max()>table[-1,3]:raise ValueError(('opening inadmissible',v.tolist(),float(p.max())))
    cfg=json.loads((opening/'launch.json').read_text());argv=cfg['argv'];i=argv.index('--root-pose')+1
    quat=Rotation.from_quat(np.array(argv[i+3:i+7],float));turn=Rotation.from_rotvec([*res.x[1:],0.]);new=(turn*quat).as_quat()
    out=foundation/'inclined-source';out.mkdir();argv[i+3:i+7]=[format(x,'.17g') for x in new]
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    data=dict(rotation_vector_rad=res.x[1:].tolist(),expected_top_before_ground_alignment_m=res.x[0],residual=v.tolist(),pressure_max_pa=p.max(),q_xyzw=new.tolist(),scope='Opening pose only. Contact areas and ground alignment must be recaptured after rotation.')
    (foundation/'inclination.json').write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__':fit(*(Path(x).resolve() for x in sys.argv[1:]))
