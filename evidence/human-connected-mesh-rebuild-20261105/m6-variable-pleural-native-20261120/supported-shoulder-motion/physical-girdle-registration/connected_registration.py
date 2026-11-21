"""Fit paired clavicle/scapula registrations on fixed skeletal frames against source joint sites and surrounding bone volumes."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from compare_registration import prepare
from compare_atlas import vertices_and_faces
from fit_source_surfaces import sample
from girdle_ik import contact_site

class Field:
    """Ray-fill a closed triangle skin on a Cartesian grid; the fit is later checked against exact Float32 faces."""
    def __init__(self, vertices, faces, h=.0015):
        self.h=h
        self.origin=vertices.min(axis=0)-3.271*h
        grid=(vertices-self.origin)/h
        shape=(np.ceil(grid.max(axis=0))+5).astype(int)
        delta=np.zeros(shape,dtype=np.int16)
        for tri in grid[faces]:
            a,b,c=tri
            u,v=b-a,c-a
            det=u[0]*v[1]-u[1]*v[0]
            if abs(det)<1e-12:continue
            lo=np.ceil(tri.min(axis=0)[:2]).astype(int)
            hi=np.floor(tri.max(axis=0)[:2]).astype(int)
            if np.any(hi<lo):continue
            x,y=np.mgrid[lo[0]:hi[0]+1,lo[1]:hi[1]+1]
            s=((x-a[0])*v[1]-(y-a[1])*v[0])/det
            t=(u[0]*(y-a[1])-u[1]*(x-a[0]))/det
            hit=(s>=0)&(t>=0)&(s+t<=1)
            z=np.ceil(a[2]+u[2]*s[hit]+v[2]*t[hit]).astype(int)
            delta[x[hit],y[hit],z]+=int(np.sign(det))
        inside=np.cumsum(delta,axis=2)!=0
        assert inside.any() and not inside[:,:,-1].any(), 'open or inconsistent voxel column'
        self.grid=((distance_transform_edt(~inside)-distance_transform_edt(inside))*h).astype(np.float32)
        self.hi=np.asarray(self.grid.shape)-1

    def values(self, points):
        x=(points-self.origin)/self.h
        k=np.all((x>=0)&(x<=self.hi),axis=1)
        d=np.full(len(points),.04)
        if np.any(k):
            d[k]=map_coordinates(self.grid,x[k].T,order=1,prefilter=False)
        return d

def points(bone,raw):
    return bone[6].apply(raw)*bone[7]+bone[8]

def run(launch,anchors,motion,archive,prior,file_out):
    cfg,cat,meshes,reg,poses,rest,data=prepare(launch,anchors,motion,archive)
    doc=json.loads(Path(cfg['argv'][3]).with_suffix('.manifest.json').read_text())
    m=next(a for a in doc['source']['anchors'] if a['member_id']=='FJ3290')
    with zipfile.ZipFile(archive) as z:
        text=z.read(next(n for n in z.namelist() if n.endswith('/FJ3290.obj')))
    assert hashlib.sha256(text).hexdigest()==m['member_sha256']
    sv,sf=vertices_and_faces(text)
    st=next(sid for sid,r in cat.items() if r['body']==m['core_body_index'] and r['vertexCount']==len(sv) and r['indexCount']==sf.size)
    assert np.array_equal(sf,meshes[st][1])
    worlds={}
    for n,(v,*_) in meshes.items():
        p=rest[cat[n]['body']]
        worlds[n]=p[:3]+R.from_quat(p[3:]).apply(v)
    stern=np.linalg.lstsq(np.c_[sv,np.ones(len(sv))],worlds[st],rcond=None)[0]
    assert np.max(np.abs(np.c_[sv,np.ones(len(sv))]@stern-worlds[st]))<1.e-6
    icp={r['sid']:r for r in json.loads(prior.read_text())}
    field={}
    def signed_field(s):
        if s not in field:field[s]=Field(worlds[s],meshes[s][1])
        return field[s]
    def pose_aff(p,body):
        a=rest[body];b=p[body]
        rr=R.from_quat(b[3:])*R.from_quat(a[3:]).inv()
        return rr, b[:3]-rr.apply(a[:3])
    results=[]; metrics=[]
    for row in reg['candidates']:
        l=row['side']==0
        scap,hum=row['stable_ids']; clav=9 if l else 8
        moving=[clav,scap];side='l' if l else 'r'
        c,s,h=(data[n] for n in [clav,scap,hum])
        sc=(contact_site(c[2],sv)+contact_site(sv,c[2]))*.5
        ac=(contact_site(c[2],s[2])+contact_site(s[2],c[2]))*.5
        gh=(contact_site(h[2],s[2])+contact_site(s[2],h[2]))*.5
        sites=[np.array([points(c,sc),points(c,ac)]),np.array([points(s,ac),points(s,gh)])]
        target=[np.r_[sc,1.]@stern,points(h,gh)]
        pivots=[p.mean(axis=0) for p in sites]
        # Scale translation parameters by 10 mm and rotation by 0.1 radians.
        def apply(z,i,x,inv=False):
            rr=R.from_rotvec(.1*z[6*i:6*i+3]);shift=.01*z[6*i+3:6*i+6];p=pivots[i]
            return rr.inv().apply(x-p-shift)+p if inv else rr.apply(x-p)+p+shift
        prior_z=[]
        for i,n in enumerate(moving):
            b=rest[cat[n]['body']];rr=R.from_quat(b[3:]);rb=R.from_quat(icp[n]['rotation_xyzw'])
            rot=rr*rb*rr.inv()
            t=rr.apply(rb.apply(rr.inv().apply(pivots[i]-b[:3]))+icp[n]['translation_m'])+b[:3]-pivots[i]
            prior_z.extend(np.r_[rot.as_rotvec()*10,t*100])
        prior_z=np.asarray(prior_z)
        sample_pts={n:sample(worlds[n],meshes[n][1],4000,527) for n in moving}
        pairs=[]
        for i,n in enumerate(moving):
            v=worlds[n]; p=apply(prior_z,i,v)
            low=np.minimum(v.min(axis=0),p.min(axis=0))-.023
            high=np.maximum(v.max(axis=0),p.max(axis=0))+.023
            for k,x in worlds.items():
                if k==n or k in moving and k<n:continue
                if np.any(x.max(axis=0)<low) or np.any(x.min(axis=0)>high):continue
                if k not in moving and np.linalg.norm((x.max(0)-x.min(0)))>.6:continue
                # Only compare reference samples near the entire attainable
                # bank. Moving banks can approach more than the sample margin.
                if k not in moving:
                    pts=sample_pts[n]
                    margin=.095
                    pts=pts[np.all((pts>x.min(0)-margin)&(pts<x.max(0)+margin),axis=1)]
                else:pts=sample_pts[n]
                if not len(pts):continue
                pairs.append((i,n,k,pts))
        batches=[]
        for i,n,k,pts in pairs:
            dg=signed_field(k);j=moving.index(k) if k in moving else -1
            for pose in (rest,poses[-3],poses[8000]):
                rn,tn=pose_aff(pose,cat[n]['body']);rk,tk=pose_aff(pose,cat[k]['body'])
                tr=rk.inv()*rn;tt=rk.inv().apply(tn-tk)
                batches.append((i,j,k,pts,tr,tt,dg))
        def contacts(z, margin, select=False):
            out=[]
            for i,j,k,pts,rr,tt,field_k in batches:
                x=rr.apply(apply(z,i,pts))+tt
                if j>=0:x=apply(z,j,x,True)
                values=field_k.values(x)
                if select:
                    # Near witnesses give a fixed smooth least-squares domain
                    # between active-set refreshes; retain spatial coverage.
                    ids=np.flatnonzero(values<margin)
                    if len(ids)>100:
                        ids=ids[np.argsort(values[ids],kind='stable')[:100]]
                    if len(ids):out.append((i,j,k,pts[ids],rr,tt,field_k))
                else:out.append(float(values.min()))
            return out
        def site_res(z):
            a=apply(z,0,sites[0]);b=apply(z,1,sites[1])
            return np.r_[a[0]-target[0],a[1]-b[0],b[1]-target[1]]*1000
        z=prior_z.copy()
        for stage in range(18):
            working=contacts(z,.008,True)
            def residual(w):
                vals=[site_res(w)*.7,(w-prior_z)*.12]
                for i,j,k,pts,rr,tt,field_k in working:
                    x=rr.apply(apply(w,i,pts))+tt
                    if j>=0:x=apply(w,j,x,True)
                    vals.append(np.minimum(0,field_k.values(x)-.0025)*1000*.65)
                return np.concatenate(vals)
            # Centered differences exceed the precision of the float32 field.
            def jac(w):
                eye=np.eye(len(w))*.001
                return np.stack([(residual(w+d)-residual(w-d))/.002 for d in eye],axis=1)
            sol=least_squares(residual,z,jac=jac,bounds=(z-.75,z+.75),max_nfev=28)
            last=z;z=sol.x
            worst=min(contacts(z,.0025))
            print(side,stage,'fit mm',np.round(np.linalg.norm(site_res(z).reshape(3,3),axis=1),2),'voxel min mm',round(worst*1000,2),flush=True)
            if np.max(abs(z-last))<.005:break
        metrics.append(dict(side=side,joint_site_errors_m=(np.linalg.norm(site_res(z).reshape(3,3),axis=1)*.001).tolist(),minimum_sampled_grid_distance_m=worst))
        for i,n in enumerate(moving):
            b=rest[cat[n]['body']];rr=R.from_quat(b[3:]);rw=R.from_rotvec(z[6*i:6*i+3]*.1)
            rb=rr.inv()*rw*rr
            t=rr.inv().apply(apply(z,i,b[:3])-b[:3])
            results.append(dict(bone=icp[n]['bone'],sid=n,body=cat[n]['body'],scale=1.,rotation_xyzw=rb.as_quat().tolist(),translation_m=t.tolist()))
        file_out.write_text(json.dumps(results,indent=2)+'\n')
        file_out.with_suffix('.metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
