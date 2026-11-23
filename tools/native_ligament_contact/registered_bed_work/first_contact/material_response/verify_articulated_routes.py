"""Test the production force-transfer and audit kernels against independent cortical paths."""
from pathlib import Path
import json,subprocess,struct,sys
import numpy as np
from scipy.spatial.transform import Rotation
from compile_connected_joint import digest
from audit_fiber_load_bounds import read_pose
from rebind_ligament_material import banks

def run(h,build,out):
    out.mkdir();ps=np.array(read_pose(h.parent/'native-ligament-material/stderr.log')['bodyPoses'])
    bh=next((h.parent/'supported-bones').glob('*.nhbones'));mesh=banks(bh)
    d=json.load(open(build/'registration.json'))
    physical=json.load(open(h/'slack-upper-bound.json'));report=[]
    for index,row in enumerate(d['records']):
        name,side=row['fiber'].split('/');section=json.load(open(h/'section-routes'/(name+'-'+side+'.json')))
        src=physical['trials'][row['fiber']]['probes'];a,b=row['owners']
        expected_active=[]
        cc=row['loaded']
        for suite in (['wide','fine','bad-tag','bad-identity'] if cc else ['wide','taut']):
            cases=src if suite=='wide' or suite=='fine' else src[:1]
            scaled=.005 if suite=='fine' else .25
            if cc and suite in ('wide','fine'):
                compare=json.load(open(h/('body-pose-'+suite)/'reference.json'))
                expected_active=[p['output'] for p in compare['records'] if p['fiber']==row['fiber']]
            at=out/(str(index)+'-'+suite);at.mkdir()
            poses=np.zeros((len(cases),len(ps),2,4),'<f4');poses[:,:,0,:3]=ps[:,:3];poses[:,:,1]=ps[:,3:]
            mats={e['body']:e for e in section['sites']};points=[]
            for body in (a,b):
                e=mats[body];face,v,_=mesh[body,e['bone']];points.extend([*(np.array(e['barycentric'])@v[face[e['triangle']]]),0.])
            K=103000. if cc else (17400. if name.startswith('superior') else 36700.)
            area=(10. if cc else (11.3 if name.startswith('superior') else 53.7))*1e-6;L=row['reference'];c3=K*L/(area*.001)
            packed=struct.pack('<4I16f',a,b,1,row['tag'] if suite!='bad-tag' else 17,*points,c3,.001,c3*.001*np.exp(.001*.05),1.05,L,area*(2 if suite=='bad-identity' else 1),1,0)
            (at/'fibers.bin').write_bytes(packed)
            J=np.zeros((len(cases),len(ps),4,3,12),'<f4')
            expected=[];audit=[]
            for env,case in enumerate(cases):
                moved=section['sites'][-1]['body']
                if suite=='taut':poses[env,moved,0,0]+=.15
                elif case['probe']!='reference':
                    mode,sign,ax=case['probe'].split('_');axis=np.eye(3)['xyz'.index(ax)]*int(sign)
                    if mode=='translate':poses[env,moved,0,:3]=ps[moved,:3]+axis*scaled*.001
                    else:poses[env,moved,1]=(Rotation.from_rotvec(axis*np.deg2rad(scaled))*Rotation.from_quat(ps[moved,3:])).as_quat()
                for k,body in enumerate((a,b)):
                    start=6*k;J[env,body,:,:,start:start+3]=np.eye(3)
                    axes=Rotation.from_quat(poses[env,body,1]).apply(np.eye(3))
                    for j,x in enumerate(axes):J[env,body,j+1,:,start+3:start+6]=np.cross(np.eye(3),x).T
                if cc and suite in ('wide','fine'):
                    f,t0,t1=np.array(expected_active[env]);expected.append(np.r_[f[:3],t0[:3],-f[:3],t1[:3]])
                    audit.append([2*t0[3],t0[3],t1[3]])
                else:expected.append(np.zeros(12));audit.append([0,0,case['length_m']/L])
            poses.tofile(at/'poses.bin');J.tofile(at/'jacobians.bin')
            subprocess.run([build/'exercise-articulated',build/'metal/NumiMatter.metallib',at],check=True)
            force=np.fromfile(at/'generalized.f32','<f4').reshape(-1,12);aa=np.fromfile(at/'audit.f32','<f4').reshape(-1,12)
            negative=suite in ('bad-tag','bad-identity','taut')
            nan=np.flatnonzero(~np.isfinite(aa).all(1)).tolist()
            diff=np.abs(np.array(expected)-force)
            ferr=float(np.max(diff[:,[0,1,2,6,7,8]]));terr=float(np.max(diff[:,[3,4,5,9,10,11]]))
            ae=None if nan else np.max(abs(aa[:,[3,7,8]]-audit),axis=0).tolist()
            ok=np.array_equal(force,np.zeros_like(force)) and len(nan)==len(cases) if negative else not nan and ferr<.01 and terr<.001 and ae[0]<.02 and ae[1]<.01 and ae[2]<3e-5 and np.max(abs(aa[:,4:7]))<.0001 and np.all(aa[:,10]==1)
            report.append(dict(fiber=row['fiber'],suite=suite,cases=len(cases),passed=bool(ok),invalid=nan,max_force_error_n=ferr,max_moment_error_nm=terr,audit_errors=ae))
            print(index,suite,ok,nan,ferr,terr,flush=True)
    result=dict(metallib=digest(build/'metal/NumiMatter.metallib'),result=report,passed=all(r['passed'] for r in report))
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result['passed']
if __name__=='__main__':sys.exit(0 if run(*(Path(s).resolve() for s in sys.argv[1:])) else 1)
