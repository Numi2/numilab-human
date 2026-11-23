"""Bound bed reaction using the actual projected area of the registered physical skin."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from fit_bed_section_curve import stress
from material_context import sha
from probe_registered_layers import measure

def submerged_fraction(z,h):
    # Linear triangle height field clipped by a horizontal half space.
    a,b,c=np.sort(z,axis=1).T
    f=np.zeros(len(z));f[c<=h]=1
    low=(a<h)&(h<b);high=(b<=h)&(h<c)&(a<h)
    f[low]=(h-a[low])**2/((b[low]-a[low])*(c[low]-a[low]))
    f[high]=1-(c[high]-h)**2/((c[high]-a[high])*(c[high]-b[high]))
    return f

def bound(case,paths,opening,capacity):
    cfg=json.loads((opening/'launch.json').read_text());curves,=[Path(x) for x in cfg['pins'] if x.endswith('/curves.json')]
    if sha(curves)!=cfg['pins'][str(curves)]:raise ValueError('Changed constitutive model')
    coeff=[(m['mu_pa'],m['log_volume_coefficient_pa']) for m in json.loads(curves.read_text())['material']['materials']]
    cap=min(stress(2/3,*c) for c in coeff);ss,sf=[brentq(lambda x:stress(x,*c)-cap,2/3,1) for c in coeff]
    cv=json.loads(capacity.read_text())
    if abs(cap-cv['pressure_ceiling_Pa'])>1e-6 or cv['curves']!=sha(curves):raise ValueError('Different comparison body')
    weight=cv['gravity_N'];areas=None;heights=None
    def sample(k,v,f):
        nonlocal areas,heights
        if k!='O':return
        points=v[f];ab=points[:,1]-points[:,0];ac=points[:,2]-points[:,0]
        areas=np.abs(ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0])/2
        heights=points[:,:,2]
    d=measure(case,paths,opening,sample)
    trial=[]
    for vertex,p in d['witnesses'].items():
        o,i,f=[p['full_volume_reference'][k][0]['dz_m'] for k in 'OID']
        if not o<i<f:raise ValueError('Unordered reference tissue')
        h=p['point'][2]+o;depth=(i-o)*(1-ss)+(f-i)*(1-sf)
        trial.append(dict(vertex=int(vertex),boundary_height_m=h,allowed_shortening_m=depth,max_descent_m=h+depth))
    h=min(t['max_descent_m'] for t in trial)
    A=lambda h:float(areas@submerged_fraction(heights,h))
    require=brentq(lambda h:cap*A(h)-weight,float(heights.min())-1e-5,float(heights.max())+1e-5)
    if cap*A(h)>=weight:verdict='bound inconclusive'
    else:verdict='insufficient support in this fixed registration and column law'
    return dict(scope='Uniform downward translation of the opening skeleton on a rigid horizontal bed. The registered outer material surface supplies the potential contact area. All triangles below the plane receive the largest admissible column pressure, including overlapping projected areas. This deliberately overestimates support. It does not bound a changing articulated posture, lateral load spreading or a deforming mattress.',
        pressure_limit_pa=cap,gravity_N=weight,candidate_material_rays=trial,largest_admissible_descent_bound_m=h,projected_area_upper_bound_m2=A(h),normal_force_upper_bound_N=cap*A(h),maximum_fraction_of_weight=cap*A(h)/weight,
        required_area_m2=weight/cap,earliest_descent_for_required_projected_area_m=require,decision=verdict,source_reference=d['physical_reference'],opening=d['target_pose_receipt'],curves=sha(curves))
if __name__=='__main__':print(json.dumps(bound(*(Path(s) for s in sys.argv[1:])),indent=2,allow_nan=False))
