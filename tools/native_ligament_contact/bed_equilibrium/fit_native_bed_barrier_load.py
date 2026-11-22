"""Choose initial physical gaps inside the existing IPC band using source P1 masses and the physical body weight."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq,root
from evaluate_registered_bed_transfer import digest

def force(gap,mass,dt,width):
    # Stationary fixed FEM plane, zero initial closing speed. Same squared-distance
    # barrier derivative and denominator floor as ipcBarrierResponse in the shipped shader.
    d=np.maximum(gap,1e-4*width);s=d*d;sh=width*width;off=s-sh
    k=10./np.maximum(dt*dt*sh/mass,1.e-20)
    return 2*d*k*(2*off*np.log(s/sh)+off*off/s)

def main(launch,pose,mom,out,placement=None):
    l=json.loads(launch.read_text());env=l['environment'];pref='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    get=lambda key:Path(env[pref+key])
    frag='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_RECORDED_ALLOCATION_'
    rows=np.loadtxt(get('TETGEN_ELEMENTS'),dtype=np.int32,skiprows=1);t=rows[:,1:5]-1;tag=rows[:,5]
    q=np.loadtxt(pose,skiprows=1)[:,1:4];ids=np.flatnonzero(q[:,2]==0)
    props={}
    if placement:
        from prepare_posterior_patch_inputs import patch
        ids,weight_prior,props=patch(launch,placement)
    else:assert len(ids)==28
    cells=np.empty(len(t))
    for name,mat in [('SKIN',15),('FAT',16)]:
        k=frag+name+'_CHILD_MASS_CSV';path=Path(env[k]);assert digest(path)==env[k+'_SHA256']
        data=np.loadtxt(path,delimiter=',',skiprows=1,usecols=(0,2))
        assert np.array_equal(data[:,0],np.arange((tag==mat).sum())) and (data[:,1]>0).all()
        cells[tag==mat]=data[:,1]
    mass=np.bincount(t.ravel(),weights=np.repeat(cells*.25,4),minlength=len(q));assert np.isclose(mass.sum(),cells.sum())
    rec=json.loads(mom.read_text());com=np.array(rec['centre_of_accounted_mass_xy_m'])
    xy=q[ids,:2];centered=xy-com;scale=np.max(np.abs(centered),axis=0);xyc=centered/scale
    def weights(coef):
        z=xyc@coef;w=(weight_prior if placement else mass[ids])*np.exp(z-z.max());return w/w.sum()
    solution=root(lambda y:weights(y)@xyc,np.zeros(2))
    assert solution.success and np.linalg.norm(weights(solution.x)@xyc)<1.e-9
    weight=float(rec['total_body_weight_at_9_81_N_kg_N'])
    dt=float(l['argv'][l['argv'].index('--muscle-step-seconds')+1]);width=float(np.float32(1.e-5))
    target=weight*weights(solution.x)
    gap=np.array([brentq(lambda g:force(g,m,dt,width)-n,.00011*width,width,xtol=2e-18) for m,n in zip(mass[ids],target)]).astype('f4').astype(float)
    predicted=force(gap,mass[ids],dt,width); assert np.isclose(predicted.sum(),weight,rtol=2e-5)
    data=dict(nodes=ids.tolist(),gaps=gap.tolist(),node_mass_kg=mass[ids].tolist(),dt_s=dt,band_m=width,minimum_m=float(gap.min()),maximum_m=float(gap.max()),
              weight_N=weight,predicted_normal_N=float(predicted.sum()),individual_force_N=predicted.tolist(),center_of_pressure_xy_m=(predicted@xy/predicted.sum()).tolist(),
              accounted_com_xy_m=com.tolist(),moment_input_sha256=digest(mom),unresolved_other_FEM_moment_mass_kg=rec['unresolved_other_FEM_moment_mass_kg'],
              patch=props,uncompressed_mass_kg=float(mass.sum()),scope='Geometry initialization at a fixed bed. Projected starting forces are never supplied to the runtime.')
    out.write_text(json.dumps(data,indent=2)+'\n');print(data['minimum_m'],data['maximum_m'],data['predicted_normal_N'])
if __name__=='__main__':main(*(Path(v).resolve() for v in sys.argv[1:]))
