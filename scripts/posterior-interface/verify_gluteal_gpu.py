"""Compare the Metal owner, including GPU scatter, with independent virtual-work derivatives."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from check_gluteal_work import evaluate


def check(parent, material, build, out):
    out.mkdir(exist_ok=True)
    z=np.load(material/'parents.npz')
    K=np.fromfile(build/'material.f32','<f4').astype(float).reshape(171,171)
    points=z['reference'][:51].astype('f4').astype(float)
    rows={k['body_index']:k['current'] for k in json.loads((parent/'native-initial-bodies.json').read_bytes())['all_bodies']}
    origins=np.array([rows[i]['position_world_m'] for i in z['donors']],dtype='f4').astype(float)
    q=np.array([rows[i]['orientation_world_xyzw'] for i in z['donors']],dtype='f4').astype(float)
    initial=Rotation.from_quat(q)
    ids=np.asarray(z['fat']+1666,dtype='<u4')
    rng=np.random.default_rng(182)
    summaries=[]
    for name,amplitude,angle in [('rest',0.,0.),('rigid',0.,1.87),('bed',1.e-4,0.),('combined',1.e-3,.37)]:
        frame=Rotation.from_rotvec([0.,angle,0.])
        drift=np.array([.12,-.07,.031]) if angle else 0.
        x=points+amplitude*(np.tile(z['bed_normal'],(51,1)) if name=='bed' else rng.normal(size=points.shape))
        o=origins.copy();r=initial
        if name=='combined':
            o+=rng.normal(size=o.shape)*amplitude
            r=Rotation.from_rotvec(rng.normal(size=q[:,:3].shape)*.014)*r
        x=frame.apply(x)+drift;o=frame.apply(o)+drift;r=frame*r
        pos=np.vstack([x,o]).astype('<f4');quat=r.as_quat().astype('<f4')
        fixture=out/(name+'.fixture');forces=out/(name+'.forces')
        fixture.write_bytes(ids.tobytes()+pos.tobytes()+quat.tobytes())
        subprocess.run([Path(__file__).with_name('test_gluteal_kernel').resolve(),build/'NumiMatter.metallib',build/'material.f32',fixture,forces],check=True)
        data=np.fromfile(forces,'<f4').reshape(65,4).astype(float)
        energy,skin,bones,torque=evaluate(pos[:51].astype(float),pos[51:].astype(float),
            Rotation.from_quat(quat.astype(float)),points,origins,initial,K)
        wanted=np.vstack([skin,np.stack([bones,torque],axis=1).reshape(6,3)])
        diff=data[:57,:3]-wanted
        errors=np.max(abs(diff),axis=1)
        obs=data[57:]
        if not np.all(np.isfinite(data)) or not np.array_equal(data[51:57,:3],obs[:6,:3]) or obs[7,3]!=1:
            raise ValueError('Missing or inconsistent GPU force observer')
        f=data[51:57:2,:3];t=data[52:57:2,:3];s=data[:51,:3]
        balance=np.linalg.norm(s.sum(0)+f.sum(0))
        moment=np.linalg.norm(np.cross(pos[:51],s).sum(0)+np.cross(pos[51:],f).sum(0)+t.sum(0))
        rel=np.linalg.norm(diff)/max(1.e-15,np.linalg.norm(wanted))
        row=dict(case=name,max_force_or_moment_error=float(errors.max()),relative_norm=float(rel),
            observed_energy_J=float(obs[7,0]),expected_energy_J=float(energy),force_balance_N=float(balance),
            moment_balance_Nm=float(moment),largest_output=float(np.max(abs(data[:57,:3]))))
        summaries.append(row)
        if errors.max()>2.e-4 or balance>2.e-5 or moment>2.e-5 or amplitude and rel>1.e-3:
            raise ValueError(row)
    (out/'gpu.json').write_text(json.dumps(summaries,indent=2)+'\n')

if __name__=='__main__':
    check(*(Path(s).resolve() for s in sys.argv[1:]))
