"""Differentiate the native compartment energy with independent finite rigid motions."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def evaluate(points, locations, orientations, ref, bases, initial, K):
    frame = orientations[0]*initial[0].inv()
    displacement = frame.inv().apply(points-locations[0])-(ref-bases[0])
    translation = frame.inv().apply(locations-locations[0])-(bases-bases[0])
    relative = frame.inv()*orientations*initial.inv()
    quat = relative.as_quat(canonical=True)
    u = np.r_[displacement.ravel(), np.c_[translation,2*quat[:,:3]].ravel()]
    g = K @ u
    grad = g[153:].reshape(3,6)
    skin = -frame.apply(g[:153].reshape(51,3))
    force = -frame.apply(grad[:,:3])
    torque = -frame.apply(quat[:,3:] * grad[:,3:]+np.cross(quat[:,:3],grad[:,3:]))
    rem_force = skin.sum(0)+force.sum(0)
    rem_torque = np.cross(points-locations[0],skin).sum(0)+np.cross(locations-locations[0],force).sum(0)+torque.sum(0)
    force[0]-=rem_force
    torque[0]-=rem_torque
    return .5*(g@u),skin,force,torque


def test(material,parent,out):
    z=np.load(material/'parents.npz');K=np.load(material/'reference-tangent.npz')['matrix']
    ref=z['reference'][:51];bases=z['origins']
    poses={b['body_index']:b['current'] for b in json.loads((parent/'native-initial-bodies.json').read_bytes())['all_bodies']}
    initial=Rotation.from_quat([poses[b]['orientation_world_xyzw'] for b in z['donors']])
    rng=np.random.default_rng(82)
    rows=[]
    for angle in (0.,.39,2.6):
        orbit=Rotation.from_rotvec(np.array([.12,-.95,.28])*angle)
        shift=np.array([-.51,.22,.34])
        points=orbit.apply(ref+rng.normal(0,4e-5,ref.shape))+shift
        locations=orbit.apply(bases+rng.normal(0,4e-5,bases.shape))+shift
        orientations=orbit*Rotation.from_rotvec(rng.normal(0,.003,(3,3)))*initial
        args=(ref,bases,initial,K)
        e,s,f,t=evaluate(points,locations,orientations,*args)
        errors=[]
        for k in range(15):
            dx=rng.normal(size=ref.shape);db=rng.normal(size=bases.shape);dw=rng.normal(size=(3,3))
            if k<3:
                dx*=0.;db*=0.;dw[np.arange(3)!=k]=0.
            eps=2e-7
            values=[evaluate(points+sign*eps*dx,locations+sign*eps*db,Rotation.from_rotvec(sign*eps*dw)*orientations,*args)[0] for sign in (1.,-1.)]
            derivative=(values[0]-values[1])/(2*eps)
            work=np.sum(s*dx)+np.sum(f*db)+np.sum(t*dw)
            errors.append(abs(derivative+work))
        balance=np.linalg.norm(s.sum(0)+f.sum(0))
        moment=np.linalg.norm(np.cross(points,s).sum(0)+np.cross(locations,f).sum(0)+t.sum(0))
        pure=evaluate(orbit.apply(ref)+shift,orbit.apply(bases)+shift,orbit*initial,*args)
        rigid_peak=max(np.max(np.abs(x)) for x in pure[1:])
        if max(errors)>1e-7 or max(balance,moment,rigid_peak)>1e-9:
            raise ValueError('Material virtual work or objectivity failed')
        rows.append(dict(rotation_rad=angle,energy_J=e,virtual_work_absolute_error=max(errors),rigid_force_peak=rigid_peak,
                         force_balance_N=balance,moment_balance_Nm=moment))
    out.write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':
    test(*(Path(s).resolve() for s in sys.argv[1:]))
