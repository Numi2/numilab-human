"""Check the source constitutive force at the observed material node by differentiating energy."""
import argparse,json
from pathlib import Path
import numpy as np
from certify_interior_motion import response
from compile_connected_joint import digest

def run(volume,source,star,out):
    measured=json.loads(star.read_bytes());r=json.loads(source.read_bytes())
    x,y=[np.loadtxt(volume/f,skiprows=1,usecols=(1,2,3)) for f in ('mesh.node','current.node')]
    node=int(measured['source']['joint_node_id'])-1
    result={}
    for k in (15,16):
        t=np.array([c['vertices'] for c in measured['cells'] if c['region']==k])-1
        vertices,t0=np.unique(t,return_inverse=True);t0=t0.reshape(t.shape)
        mat=r['materials'][str(k)]['registered'];mu,bulk=mat['source_mu_pa'],mat['source_logJ_coefficient_pa']
        n=np.flatnonzero(vertices==node).item();X=x[vertices];Y=y[vertices].copy()
        h=1e-9;fd=[]
        for dim in range(3):
            v=Y.copy();v[n,dim]+=h
            plus=response(X,v,t0,mu,bulk)[0];v[n,dim]-=2*h
            minus=response(X,v,t0,mu,bulk)[0];fd.append((minus-plus)/(2*h))
        target=np.array(mat['witness_material_internal_force_N'])
        err=np.linalg.norm(fd-target)/np.linalg.norm(target)
        if err>1e-5:raise ValueError('Failed energy derivative '+str(err))
        result[k]=dict(analytic_N=target.tolist(),finite_difference_N=fd,relative_error=float(err))
    out.write_text(json.dumps(dict(star=digest(star),reference=digest(volume/'mesh.node'),registered=digest(volume/'current.node'),difference_half_width_m=h,materials=result),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('volume','source','star','out'):p.add_argument(n,type=Path)
    run(**vars(p.parse_args()))
