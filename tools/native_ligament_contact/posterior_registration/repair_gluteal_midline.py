"""Fit one material-coordinate patch on each gluteus across source and registered poses."""
import json,struct,sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix,eye
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from bilateral_bindings import decode
from solve_patch_qp import solve
from fit_bilateral_gluteal_bindings import pair_count
from prepare_posterior_muscles import load_registered,interface_audit,mesh_volume,closed
from compare_hand_interfaces import query

def source_axis(payload,sid,poses,weight):
    raw=payload.read_bytes();_,_,nr,nb,*_=struct.unpack_from('<8s6I32s',raw)
    rows=np.frombuffer(raw,'<u4',nr*8,64).reshape(-1,8)
    rec=next(r for r in rows if r[6]==sid)
    axes=[]
    for j in range(rec[1]):
        body,*t=struct.unpack_from('<I8f',raw,64+nr*32+36*(int(rec[0])+j))
        axis=Rotation.from_quat(t[3:7]).apply([t[7],0.,0.])
        axes.append(Rotation.from_quat(poses[body,3:]).apply(axis))
    return weight@np.array(axes)
def volume_gradient(v,f,axis):
    x=v.astype(float)-v.mean(0)
    g=np.zeros_like(x)
    for i in range(3):
        np.add.at(g,f[:,i],np.cross(x[f[:,(i+1)%3]],x[f[:,(i+2)%3]])/6)
    return np.einsum('ij,ij->i',g,axis)
def run(w,cap,package,audit,exe,folder):
    meshes,_,identity=load_registered(w,cap,package,audit)
    args=json.loads(package.read_text())['argv'];payload=Path(args[args.index('--soft-tissue-payload')+1])
    poses=np.array(next(json.loads(row.partition('=')[2]) for row in (cap/'emitted-native/stderr.log').open() if row.startswith('registered_ligament_actual_pose_json='))['bodyPoses'])
    data=decode(payload,[29,30],poses)
    center=(poses[131,:3]+poses[145,:3])/2
    axis=poses[145,:3]-poses[131,:3];axis/=np.linalg.norm(axis)
    anchors=json.loads((folder/'binding-parity.json').read_text());prepared={};diagnostics={}
    for sid in (29,30):
        sign=-1 if sid==29 else 1
        key=(51005,sid);native,f=meshes[key];v=native.astype(float)
        src,term,weights,bodies,_=data[sid];src=src.astype(float)
        local_axis=np.array([float(sign),0.,0.])
        derivative=sign*source_axis(payload,sid,poses,weights)
        slopes=derivative@(sign*axis)
        if np.any(slopes<.5):raise ValueError('Source-axis skinning map is not monotone')
        cv=(v-center)@(sign*axis);cs=src@local_axis
        base=json.loads((folder/str(sid)/'intersections.json').read_text())
        ids=np.unique(f[np.unique(np.array(base['51005/%d'%(59-sid)])[:,0])])
        sites=anchors[str(sid)]['actuator_sites_world_m']
        pin=np.unique(np.r_[np.flatnonzero(weights.max(1)>.999999),
              *[f[np.any(np.isin(f,cKDTree(term[:,j]).query(np.array([z for k,z in sites.items() if k.endswith(tag)]))[1]),axis=1)].ravel() for j,tag in enumerate(('origin','insertion'))]])
        dist=cKDTree(v[ids]).query(v)[0]
        # Only extend freedom to the nearest intact surface ring surrounding the conflicting faces.
        free=np.flatnonzero((dist<.05)&~np.isin(np.arange(len(v)),pin))
        lo=np.full(len(v),-.010)
        near=dist<.012
        for depth,response in ((cs,np.ones(len(cs))),(cv,slopes)):
            # Keep a strictly sloped medial contour and numerical separation at the furthest tip.
            desired=.00002+.38*(depth[near]-depth[near].min())
            lo[near]=np.maximum(lo[near],(desired-depth[near])/response[near])
        failure=np.flatnonzero((lo>0)&~np.isin(np.arange(len(v)),free))
        diagnostics[sid]=dict(free=len(free),anchored=len(pin),required_pinned=len(failure),largest_required_pinned_m=float(max(lo[failure],default=0)))
        if len(failure):
            # Actual attachment neighborhoods are fixed. Keep the exact witnesses if a proposed separator crosses one.
            diagnostics[sid]['required_pinned_vertices']=failure.tolist()
        e=np.unique(np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
        i,j=e.T;ell=np.maximum(.0005,np.linalg.norm(v[i]-v[j],axis=1))
        # Minimize relative edge shear with a bounded displacement term.
        h=(.006/ell)**2
        H=coo_matrix((np.r_[h,h,-h,-h],(np.r_[i,j,i,j],np.r_[i,j,j,i])),shape=(len(v),len(v))).tocsr()+.07*eye(len(v))
        H=H[free][:,free]
        # The source volume is affine in a single shared source-axis displacement.
        # The posed volume uses its current directional derivative and is checked exactly after emission.
        g0=volume_gradient(src,f,np.tile(local_axis,(len(v),1)))[free];g1=volume_gradient(v,f,derivative)[free]
        G=np.vstack([g0/max(abs(g0)),g1/max(abs(g1))])
        lower=lo[free]*1000;upper=np.full(len(free),15.)
        opt=minimize(lambda q:(.5*q@(H@q),H@q),np.clip(np.zeros(len(free)),lower,upper),jac=True,method='SLSQP',
                     bounds=list(zip(lower,upper)),constraints=[dict(type='eq',fun=lambda u:G@u,jac=lambda u:G)],
                     options=dict(ftol=1e-8,maxiter=240))
        q,iterations=solve(H,G,lower,upper,opt.x)
        diagnostics[sid]['active_set_polish_iterations']=iterations
        delta=np.zeros(len(v));delta[free]=q*.001
        emitted=(src+delta[:,None]*local_axis).astype('f4')
        dp=sign*(emitted[:,0]-src[:,0])
        assert np.all(dp[pin]==0)
        posed=(v+derivative*dp[:,None]).astype('f4')
        prepared[sid]=(emitted,posed,f)
        diagnostics[sid].update(max_displacement_m=float(abs(dp).max()),fixed_vertices=len(v)-len(free),
           source_volume_fraction=mesh_volume(emitted,f)/mesh_volume(src,f),posed_volume_fraction=mesh_volume(posed,f)/mesh_volume(v,f))
        np.savez_compressed(folder/str(sid)/'midline-source-patch.npz',indices=np.flatnonzero(dp),source_x=emitted[dp!=0,0],fixed_site_vertices=pin)
    def count(a,b):return pair_count(exe,prepared[29][a],prepared[29][2],prepared[30][b],prepared[30][2])
    post=dict(meshes);out={}
    for sid in (29,30):post[51005,sid]=prepared[sid][1:]
    for sid in (29,30):
        new=interface_audit((51005,sid),post,exe);old=json.loads((folder/str(sid)/'intersections.json').read_text())
        def keys(d):return {(k,i,j) for k,rr in d.items() for i,j in rr}
        a,b=keys(old),keys(new)
        out[sid]=dict(after={k:len(p) for k,p in new.items()},new=len(b-a),cleared=len(a-b),self_pairs=len(new.get('51005/%d'%sid,[])))
        (folder/str(sid)/'midline-witnesses.json').write_text(json.dumps(new,separators=(',',':'))+'\n')
    report=dict(source=identity,fit=diagnostics,registered_interfaces=out,source_pairs=count(0,0),posed_pairs=count(1,1))
    (folder/'midline-fit.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
