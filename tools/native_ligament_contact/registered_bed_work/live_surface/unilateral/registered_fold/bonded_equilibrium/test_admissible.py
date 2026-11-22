"""Compare bonded descent with independently differenced energy and Float32 mass accounting."""
import numpy as np
from scipy.linalg import null_space
from admissible import material_direction, emit_conservative
from refine_posterior_contact import mechanical

def test():
    x=np.array([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.],[0.,0.,.007],[.001,.001,-.004]])
    cells=np.array([[0,1,2,3],[0,2,1,4]])
    material=np.array([15,16]);props=np.array([[36000.,80000.],[5500.,75000.]])
    free=np.array([0,1,2]);w=np.array([[.004,.003,.002]])
    C=np.kron(w/np.linalg.norm(w),np.eye(3));rng=np.random.default_rng(294)
    worst=0.;worst_moment=0.;decreased=0
    for _ in range(20):
        y=x+rng.normal(0,.00015,x.shape)
        en,forces=mechanical((x,y),cells,material,props)
        d=material_direction(x,y,cells,free,forces[free],C,trust_um=15)
        if np.linalg.norm(C@d.ravel())>1e-12:raise ValueError('Moment changed during solve')
        power=np.sum(d*forces[free]); eps=1e-3
        if not power>0:raise ValueError('No descent')
        def E(fraction):
            q=y.copy();q[free]+=fraction*d
            D=np.linalg.solve(x[cells[:,1:]]-x[cells[:,:1]],q[cells[:,1:]]-q[cells[:,:1]])
            det=np.linalg.det(D);vol=np.linalg.det(x[cells[:,1:]]-x[cells[:,:1]])/6
            return vol@(.5*props[:,0]*((D*D).sum(axis=(1,2))-3-2*np.log(det))+.5*props[:,1]*np.log(det)**2)
        finite=(E(eps)-E(-eps))/(2*eps);worst=max(worst,abs(finite+power)/power)
        if E(1)<E(0):decreased+=1
        # A translated configuration has nonuniform output quanta.
        base=(y+np.array([.02,-.72,.03])).astype('f4').astype(float);p=base.copy();p[free]+=d
        q=emit_conservative(p,base,free,w)
        worst_moment=max(worst_moment,float(abs(w@(q[free]-base[free])).max()))
    assert decreased==20 and worst<2e-6 and worst_moment<=8e-11
    return dict(cases=20,energy_decreases=decreased,maximum_relative_work_error=worst,maximum_emitted_moment_drift_kg_m=worst_moment)
if __name__=='__main__':
    import json
    print(json.dumps(test(),indent=2))
