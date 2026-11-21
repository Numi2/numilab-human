"""Compare actual production Metal ligament audits with independent constrained solves."""
import json,sys,struct,subprocess,tempfile
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,nnls
from scipy.spatial.transform import Rotation as R

def main(root):
    routes=json.loads((root/'strip.json').read_text())['routes']
    results=[]
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)
        for s in routes:
            add=lambda x,w=0:struct.pack('<4f',*x,w)
            anchor=np.array(s['anchor']);pivot=np.array(s['pivot']);pr=np.array(s['prefixPoints'])
            incoming=pr[-1]-pr[-2];incoming/=np.linalg.norm(incoming)
            edges=s['edges'];verts=np.array([e['vertices'] for e in edges]);base=verts[:,0,:]
            delta=verts[:,1,:]-base
            # Use a 1.01 test prestrain to load both rows and exercise nonzero balance.
            area=1e-5;c4=.001;ref=s['reference'];c3=17400*ref/(area*c4);c5=c3*c4*np.exp(c4*.05)
            row=struct.pack('<4I',0,1,3,1)+add(anchor)+add(s['target'])+add([c3,c4,c5],1.05)+add([ref,area,1.01])
            side=struct.pack('<4I',len(edges),len(s['pivotNormals']),0,1)
            side+=add(pivot-anchor,s['prefix'])+add(incoming)+add(s['terminalNormal'])
            side+=b''.join(add(n) for n in s['pivotNormals'])+bytes(16*(12-len(s['pivotNormals'])))
            for e in edges:
                v=np.array(e['vertices'])-anchor
                side+=add(v[0],e['t'])+add(v[1])+add(e['normal'][0])+add(e['normal'][1])
            side+=bytes(64*(8-len(edges)));(p/'row').write_bytes(row+side)
            rng=np.random.default_rng(744+s['side']);records=[]; jacobians=[]; expected_wrenches=[]
            target=np.array(s['terminal']);normal=np.array(s['terminalNormal'])
            def reference(x):
                def measure(t):
                    points=np.vstack([pivot,base+t[:,None]*delta,x]);d=np.diff(points,axis=0)
                    lengths=np.linalg.norm(d,axis=1);u=d/np.maximum(lengths[:,None],1.e-16)
                    return lengths.sum()*1000, ((u[:-1]-u[1:])*delta).sum(1)*1000
                sol=minimize(measure,[e['t'] for e in edges],jac=True,method='L-BFGS-B',bounds=[(0,1)]*len(edges),
                             options={'ftol':1e-15,'gtol':1e-10,'maxiter':300})
                pts=np.vstack([pivot,base+sol.x[:,None]*delta,x]);d=np.diff(pts,axis=0);u=d/np.maximum(np.linalg.norm(d,axis=1)[:,None],1.e-16)
                good=all((0<t<1) for t in sol.x) and nnls(np.array(s['pivotNormals']).T,incoming-u[0])[1]<3e-5
                for i,e in enumerate(edges):
                    good=good and nnls(np.array(e['normal']).T,u[i]-u[i+1])[1]<3e-5
                return s['prefix']+sol.fun/1000, good and -u[-1]@normal>=-3e-5, pts[-2], u[-1]
            for j in range(42):
                d=np.zeros(3) if j==0 else rng.normal(size=3)*([.000005,.00003,.00015,.001][(j-1)%4])
                world=R.from_rotvec(rng.normal(size=3)*.7);origin=rng.uniform(-.7,.7,3)
                point=target+d
                normalRotation=R.from_rotvec(np.array([np.pi,0.,0.])) if j==41 else R.identity()
                pose=[[*origin,0,*world.as_quat()],[*(origin+world.apply(point-normalRotation.apply(s['target']))),0,*(world*normalRotation).as_quat()]]
                L,ok,last,unit=reference(point)
                # Each DOF supplies the velocity of the COM and its three body-axis probes.
                jac=np.zeros((2,4,3,12))
                for body,rot in enumerate((world,world*normalRotation)):
                    jac[body,:,:,6*body:6*body+3]=np.eye(3)
                    for k,axis in enumerate(rot.apply(np.eye(3))):
                        jac[body,1+k,:,6*body+3:6*body+6]=np.cross(np.eye(3),axis).T
                jacobians.append(jac)
                uu=world.apply(unit)
                expected_wrenches.append(np.r_[uu,np.cross(world.apply(last),uu),-uu,
                    np.cross((world*normalRotation).apply(s['target']),-uu)])
                if j==41: ok=False
                records.append((pose,L,ok))
            np.asarray([r[0] for r in records],'<f4').tofile(p/'poses')
            np.asarray(jacobians,'<f4').tofile(p/'jac')
            subprocess.run([root/'audit_gpu',root/'build/NumiMatter.metallib',p/'row',p/'poses',p/'audit',p/'jac',p/'force'],check=True)
            a=np.fromfile(p/'audit','<f4').reshape(-1,12);valid=np.isfinite(a).all(axis=1)
            wanted=np.array([r[2] for r in records])
            L=np.array([r[1] for r in records]);stretch=L/ref*1.01
            expected=c3*np.expm1(c4*(np.clip(stretch,1,1.05)-1))+c5*np.maximum(stretch-1.05,0)
            expected*=area
            force=np.fromfile(p/'force','<f4').reshape(-1,12)
            wantforce=expected[:,None]*np.array(expected_wrenches)
            err=abs(force[valid]-wantforce[valid])
            assert not np.any(force[~valid]) and not np.any(force[-1])
            record=dict(max_generalized_force_error_N=float(err[:,[0,1,2,6,7,8]].max()),max_generalized_moment_error_Nm=float(err[:,[3,4,5,9,10,11]].max()),side=s['side'],cases=len(valid),accepted=int(sum(valid)),expected_accepted=int(sum(wanted)),
                disagreements=np.flatnonzero(valid!=wanted).tolist(),false_accepts=np.flatnonzero(valid&~wanted).tolist(),
                max_length_error_m=float(max(abs(a[valid,8]/1.01*ref-L[valid]))),
                max_tension_error_N=float(max(abs(a[valid,7]-expected[valid]))),
                max_moment_error_Nm=float(max(np.linalg.norm(a[valid,4:7],axis=1))),
                max_force_error_N=float(max(np.linalg.norm(a[valid,:3],axis=1))))
            results.append(record)
    print(json.dumps(results,indent=2));(root/'gpu-parity.json').write_text(json.dumps(results,indent=2)+'\n')
    assert all(not r['disagreements'] and r['max_length_error_m']<2e-7 and r['max_generalized_force_error_N']<.006 and r['max_generalized_moment_error_Nm']<.001 and r['max_tension_error_N']<.005 and r['max_moment_error_Nm']<5e-5 for r in results)
if __name__=='__main__':main(Path(__file__).resolve().parent)
