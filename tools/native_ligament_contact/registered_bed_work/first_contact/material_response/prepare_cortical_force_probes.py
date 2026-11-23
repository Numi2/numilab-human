"""Prepare actual entheses, cortical edge geometry and independent reference wrenches for NumiMatter."""
from pathlib import Path
import json,sys,struct
import numpy as np
from scipy.spatial.transform import Rotation
from rebind_ligament_material import banks
from audit_fiber_load_bounds import read_pose
from compile_connected_joint import digest

def run(parent,routes,probes,change_source,out):
    d=json.loads(probes.read_text());bone=next((parent/'supported-bones').glob('*.nhbones'))
    log=parent/'native-ligament-material/stderr.log';poses=read_pose(log)['bodyPoses']
    if digest(bone)!=d['bones'] or digest(log)!=d['pose']:raise ValueError('Unmatched physical source')
    bk=banks(bone);laws=json.loads((parent/'registered-ligaments/registration.json').read_text())['fibers']
    old=json.loads(change_source.read_text())
    if old['bones']!=d['bones'] or old['pose']!=d['pose']:raise ValueError('Contact-switch source differs')
    out.mkdir();records=[];ranges=[];E=[];P=[];fibers=[]
    def f4(v,w=0.):return [*map(float,v),float(w)]
    for name,row in d['trials'].items():
        if not name.startswith('coracoclavicular_complex/'):continue
        r=json.loads((routes/(name.replace('/','-')+'.json')).read_text());law=laws[name];p0,p1=law['endpoints'];b0=p0['body'];b1=p1['body'];fp=np.array(poses[b0]);sp=np.array(poses[b1]);R=Rotation.from_quat(fp[3:]);S=Rotation.from_quat(sp[3:])
        y=np.array(r['world_m']);a=R.inv().apply(y[0]-fp[:3]);b=S.inv().apply(y[-1]-sp[:3])
        f,v,_=bk[b0,p0['bone']];n=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
        n*=np.sign(np.einsum('ij,ij->',n,v[f[:,0]]-v.mean(0)));n/=np.maximum(1e-30,np.linalg.norm(n,axis=1))[:,None]
        K=law['stiffness_n_per_m'];area=law['area_m2'];slack=r['slack_m'];c4=.001;lmax=1.05;c3=K*slack/(area*c4);c5=c3*c4*np.exp(c4*(lmax-1))
        alternates={tuple(p['added_feature']['source_edge']):p['added_feature'] for p in old['trials'][name]['probes'] if p.get('added_feature')}
        if len(alternates)>1:raise ValueError('Need more than one alternate contact region')
        options=[]
        for alt in [None,*alternates.values()]:
            chain=r['edges']+([alt] if alt else [])
            offset=len(E)
            for e in chain:
                A,B=v[e['source_edge']]-a;i,j=e['incident_faces']
                E.append([f4(A,e.get('weight',.5)),f4(B),f4(n[i]),f4(n[j])])
            # The departure test is a convex wedge test. Check both opposing
            # face vertices against the other's outward half-space first.
            last=chain[-1];i,j=last['incident_faces'];anchor=v[last['source_edge'][0]]
            if max(((v[f[i]]-anchor)@n[j]).max(),((v[f[j]]-anchor)@n[i]).max())>2e-8:
                raise ValueError('Concave or inconsistently oriented departure edge')
            options.extend([offset,len(chain)])
        if len(options)==2:options.extend([0,0])
        for sample in row['probes']:
            if not sample['passes'] or not sample['loaded']:raise ValueError('Unqualified target for force certification')
            ranges.append(options)
            target=sp.copy();label=sample['probe']
            if label!='reference':
                mode,sign,xyz=label.split('_');ax=np.eye(3)['xyz'.index(xyz)]*int(sign)
                if mode=='translate':target[:3]+=ax*d['translation_mm']*.001
                elif mode=='rotate':target[3:]=(Rotation.from_rotvec(ax*np.deg2rad(d['rotation_deg']))*S).as_quat()
            P.extend([f4(fp[:3]),fp[3:],f4(target[:3]),target[3:]])
            fibers.append(struct.pack('<4I16f',b0,b1,1,0,*f4(a),*f4(b),c3,c4,c5,lmax,slack,area,1,0))
            stretch=sample['length_m']/slack
            tension=area*(c3*np.expm1(c4*(min(stretch,lmax)-1))+c5*max(stretch-lmax,0))
            unit=np.array(sample['free_span_direction']);force=unit*tension
            arm2=np.array(sample['target_moment_arm_mm'])*.001
            # Equal and opposite total wrenches with the correct moment about each body's own COM.
            tau2=-arm2*tension;tau1=-tau2+np.cross(target[:3]-fp[:3],force)
            records.append(dict(fiber=name,probe=label,primary_edges=options[1],alternate_edges=options[3],expected_range=options[2] if sample.get('added_feature') else options[0],output=[f4(force,sample['length_m']),f4(tau1,tension),f4(tau2,stretch)]))
    (out/'fibers.bin').write_bytes(b''.join(fibers));np.array(ranges,'<u4').tofile(out/'ranges.bin');np.array(E,'<f4').tofile(out/'edges.bin');np.array(P,'<f4').tofile(out/'poses.bin')
    (out/'reference.json').write_text(json.dumps(dict(bones=digest(bone),poses=digest(log),probe_source=digest(probes),records=records),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
