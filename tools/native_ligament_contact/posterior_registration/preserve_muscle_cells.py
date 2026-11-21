"""Restore valid orientation in a source-space fit using the complete muscle volume."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,csr_matrix,eye,hstack,vstack
from scipy.spatial import cKDTree
from bilateral_bindings import decode
from material_context import sha
from prepare_posterior_muscles import load_registered,interface_audit,mesh_volume
from repair_gluteal_midline import volume_gradient,source_axis
from fit_bilateral_gluteal_bindings import pair_count
from compare_hand_interfaces import query

def tetrahedra(poly,v,f,tg):
    with poly.open('w') as fp:
        print(len(v),3,0,0,file=fp)
        for i,p in enumerate(v):print(i,*[format(x,'.17g') for x in p],file=fp)
        print(len(f),1,file=fp)
        for i,t in enumerate(f):print(1,0,i+1,file=fp);print(3,*t,file=fp)
        print(0,file=fp);print(0,file=fp)
    subprocess.run([tg,'-pzYQf',poly],check=True,stdout=subprocess.DEVNULL)
    def load(ext):return np.loadtxt(poly.with_suffix('.1.'+ext),skiprows=1)
    x=load('node')[:,1:4]
    if not np.array_equal(x,v):raise ValueError('Unmapped volume node')
    raw=load('face').astype(int);marked=raw[raw[:,-1]>0];order=marked[:,4].argsort()
    if not np.array_equal(np.sort(marked[order,1:4],axis=1),np.sort(f,axis=1)):raise ValueError('Boundary ancestry mismatch')
    t=load('ele')[:,1:5].astype(int)
    if det(v,t).min()<=0:raise ValueError('Initial cell orientation')
    return t
def det(v,t):
    a,b,c,d=v[t].transpose(1,0,2)
    return np.einsum('ij,ij->i',b-a,np.cross(c-a,d-a))
def det_derivative(v,direction,t):
    a,b,c,d=v[t].transpose(1,0,2)
    g=np.stack((np.cross(c-a,d-a),np.cross(d-a,b-a),np.cross(b-a,c-a)),axis=1)
    g=np.concatenate((-g.sum(1)[:,None,:],g),axis=1)
    data=np.einsum('tij,tij->ti',g,direction[t])/det(v,t)[:,None]*.001
    return coo_matrix((data.ravel(),(np.repeat(np.arange(len(t)),4),t.ravel())),shape=(len(t),len(v))).tocsr()
def pair_axis(A,B):
    a=np.roll(A,-1,axis=0)-A;b=np.roll(B,-1,axis=0)-B
    na=np.cross(a[0],a[1]);nb=np.cross(b[0],b[1])
    cand=np.vstack((na,nb,np.cross(a[:,None,:],b).reshape(-1,3),np.cross(na,a),np.cross(nb,b)))
    lens=np.linalg.norm(cand,axis=1);cand=cand[lens>1e-19]/lens[lens>1e-19,None]
    cand=np.concatenate((cand,-cand));proj=(B[:,None,:]-A[None,:,:]).reshape(9,3)@cand.T
    best=proj.min(0).argmax();n=cand[best];sep=proj[:,best].min()
    return None if sep<-1e-10 else (n,sep)
def sat_pair(x,direction,f,i,j):
    choice=pair_axis(x[f[i]],x[f[j]])
    if choice is None:return None
    n,sep=choice;rows=[];bounds=[]
    for ia in f[i]:
        for ib in f[j]:
            if ia==ib:continue
            r=np.zeros(len(x));r[ia]=direction[ia]@n*.001;r[ib]=-direction[ib]@n*.001
            gap=(x[ib]-x[ia])@n
            rows.append(r);bounds.append(gap-min(1e-7,max(0.,sep)*.25))
    return csr_matrix(np.array(rows)),np.array(bounds)
def run(w,cap,package,audit,exe,tetgen,folder,guard=False):
    meshes,_,identity=load_registered(w,cap,package,audit);cfg=json.loads(package.read_text())['argv']
    payload=Path(cfg[cfg.index('--soft-tissue-payload')+1])
    poses=np.array(next(json.loads(row.partition('=')[2]) for row in (cap/'emitted-native/stderr.log').open() if row.startswith('registered_ligament_actual_pose_json='))['bodyPoses'])
    center=(poses[131,:3]+poses[145,:3])/2
    lateral=poses[145,:3]-poses[131,:3];lateral/=np.linalg.norm(lateral)
    data=decode(payload,[29,30],poses);fitted={};stats={};source_points={}
    for sid in (29,30):
        local=folder/str(sid)
        v,faces=meshes[51005,sid];v=v.astype(float)
        src,terms,weights,bodies,_=data[sid];src=src.astype(float)
        sign=1 if sid==30 else -1
        sd=np.tile([sign,0.,0.],(len(v),1));nd=sign*source_axis(payload,sid,poses,weights)
        st=tetrahedra(local/'reference.poly',src,faces,tetgen)
        nt=np.loadtxt(local/'source.1.ele',skiprows=1,dtype=int)[:,1:5]
        shape=((src,sd,st),(v,nd,nt))
        depths=[sign*src[:,0],(v-center)@(sign*lateral)];slopes=[np.ones(len(v)),nd@(sign*lateral)]
        previous=np.load(local/'midline-source-patch.npz')
        q0=np.zeros(len(v));ids=previous['indices'];q0[ids]=sign*(previous['source_x']-src[ids,0])*1000
        pins=previous['fixed_site_vertices']
        dbase=json.loads((local/'intersections.json').read_text())
        hit=np.unique(faces[np.unique(np.array(dbase['51005/%d'%(59-sid)])[:,0])])
        dist=cKDTree(v[hit]).query(v)[0];free=np.flatnonzero((dist<.057)&~np.isin(np.arange(len(v)),pins))
        n=len(free);lo=np.full(n,-10.)
        # Preserve the separating side already reached by the joined fit in both frames.
        for depth,slope in zip(depths,slopes):
            reached=depth+slope*q0*.001
            need=np.minimum(reached,.00002)
            close=dist[free]<.013
            lo[close]=np.maximum(lo[close],1000*(need[free][close]-depth[free][close])/slope[free][close])
        eyeN=eye(n,format='csr');G=[];rhs=[]
        for x,d,t in shape:
            D=det_derivative(x,d,t)[:,free];used=np.flatnonzero(D.getnnz(axis=1))
            G.append(-D[used]);rhs.append(np.full(len(used),.80))
        grad=np.stack([volume_gradient(x,faces,d)[free] for x,d,_ in shape]);grad/=np.abs(grad).max(1)[:,None]
        constraints=[];history=[];seen=set()
        prefix='guarded' if guard else 'cell-fit'
        for generation in range(12):
            F=vstack(G+[c[0] for c in constraints],format='csr')
            bounds=np.concatenate(rhs+[c[1] for c in constraints])
            AA=vstack((hstack((F,csr_matrix((F.shape[0],n)))),hstack((eyeN,-eyeN)),hstack((-eyeN,-eyeN))),format='csr')
            eq=hstack((csr_matrix(grad),csr_matrix((2,n))))
            result=linprog(np.r_[np.zeros(n),np.ones(n)],A_ub=AA,b_ub=np.r_[bounds,q0[free],-q0[free]],A_eq=eq,b_eq=[0.,0.],bounds=list(zip(lo,np.full(n,15.)))+[(0,None)]*n,method='highs')
            if not result.success:
                history.append(dict(failed=result.message));break
            q=np.zeros(len(v));q[free]=result.x[:n]*.001
            ref=(src+sd*q[:,None]).astype('f4');q=sign*(ref[:,0]-src[:,0])
            posed=(v+nd*q[:,None]).astype('f4')
            actual=[ref,posed];trial=[];witnesses=[]
            for k,((x,d,t),xx) in enumerate(zip(shape,actual)):
                xx=xx.astype(float);ratio=det(xx,t)/det(x,t)
                changed=np.any(q[faces]!=0,axis=1)
                crosses=query(exe,xx,faces,changed)
                bad=int(np.sum(ratio<=0))
                trial.append(dict(minimum_ratio=float(ratio.min()),inverted=bad,self_pairs=len(crosses),volume_fraction=mesh_volume(xx,faces)/mesh_volume(x,faces)))
                for i,j in sorted(crosses):
                    sep=sat_pair(x,d,faces,i,j)
                    if sep is not None:witnesses.append((sep[0][:,free],sep[1]))
                # Correct the nonlinear posed-volume mapping about the new iterate.
                badids=np.flatnonzero(ratio<.08)
                if len(badids):
                    D=det_derivative(xx,d,t[badids]).multiply(ratio[badids,None]).tocsr()[:,free]
                    witnesses.append((-D,ratio[badids]-.12-D@(q[free]*1000)))
            excess={}
            if guard and not witnesses:
                current=dict(meshes);current[51005,sid]=(posed,faces)
                banks=interface_audit((51005,sid),current,exe)
                for key,rows in banks.items():
                    sem,other=map(int,key.split('/'))
                    if sem==51005 and other in (29,30):continue
                    old=set(map(tuple,dbase.get(key,[])))
                    critical=sem in (51004,51007)
                    if len(rows)<=len(old) and not critical:continue
                    if len(rows)>len(old):excess[key]=len(rows)-len(old)
                    tv,tf=meshes[sem,other]
                    for i,j in set(map(tuple,rows))-old:
                        if (key,i,j) in seen:continue
                        seen.add((key,i,j));A=v[faces[i]];B=tv[tf[j]].astype(float)
                        plane=pair_axis(A,B)
                        if plane is None:continue
                        normal,sep=plane
                        cols=faces[i]
                        # This opposing anatomical surface retains its source coordinates.
                        mat=coo_matrix((nd[cols]@normal*.001,(np.arange(3),cols)),shape=(3,len(v))).tocsr()
                        bound=(B@normal).min()-A@normal-min(1e-7,max(0.,sep)*.25)
                        witnesses.append((mat[:,free],bound))
                trial[-1]['excess_against_neighbor_banks']=excess
            history.append(trial)
            if not witnesses and all(row['inverted']==0 and row['self_pairs']==0 for row in trial):break
            if not witnesses:break
            constraints.extend(witnesses)
        if not history or not isinstance(history[-1],list):stats[sid]=dict(trials=history);continue
        stats[sid]=dict(trials=history,pinned=len(pins),physically_oriented=all(z['inverted']==0 and z['self_pairs']==0 for z in trial))
        np.savez_compressed(local/(prefix+'.npz'),indices=np.flatnonzero(q),source_x=ref[q!=0,0],fixed_site_vertices=pins)
        fitted[sid]=(posed,faces);source_points[sid]=(ref,faces)
    report=dict(source=identity,sides=stats)
    if len(fitted)==2:
        newmeshes=dict(meshes)
        for sid,vf in fitted.items():newmeshes[51005,sid]=vf
        report['source_pair_count']=pair_count(exe,*source_points[29],*source_points[30])
        report['registered_pair_count']=pair_count(exe,*fitted[29],*fitted[30])
        for sid in fitted:
            new=interface_audit((51005,sid),newmeshes,exe)
            (folder/str(sid)/(prefix+'-witnesses.json')).write_text(json.dumps(new,separators=(',',':'))+'\n')
            stats[sid]['crossing_counts']={k:len(x) for k,x in new.items()}
    (folder/(prefix+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print([(k,v['trials'][-1]) for k,v in stats.items()])
if __name__=='__main__':
    values=sys.argv[1:];guard=values[-1]=='--guard'
    run(*map(Path,values[:-1] if guard else values),guard=guard)
