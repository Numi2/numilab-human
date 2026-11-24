"""Correct cortical penetration in the registered tissue pose under physical element constraints."""
from pathlib import Path
import ctypes,json,sys
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix,hstack,vstack,eye
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from supported_material_admission import geom
from rebind_ligament_material import banks
from compare_supported_postures import poses
from inspect_shoulder_source import nearest
from compile_connected_joint import digest

def gradients(x):
    a,b,c=(x[:,i]-x[:,0]for i in (1,2,3))
    g=np.stack([np.cross(b,c),np.cross(c,a),np.cross(a,b)],axis=1)
    return np.concatenate([-g.sum(1)[:,None,:],g],axis=1)

def fit(y,x,T,bad,delta,anchors,protected,spacing,lib):
    # Compact trilinear functions act as a reduced construction basis. Their
    # values move the actual skin/fat nodes in the shared native configuration.
    ij=np.unique(np.floor(y[bad]/spacing).astype(int),axis=0)
    corners=np.indices((2,2,2)).reshape(3,8).T
    control=np.unique((ij[:,None]+corners[None]).reshape(-1,3),axis=0)
    control=np.array([c for c in control if np.min(np.max(abs(anchors-c*spacing),axis=1),initial=np.inf)>=spacing])
    lookup={tuple(c):i for i,c in enumerate(control)}
    regions=np.flatnonzero(((y>=control.min(0)*spacing-spacing)&(y<=control.max(0)*spacing+spacing)).all(1))
    v=y[regions]/spacing;grid=np.floor(v).astype(int);frac=v-grid;rr=[];cc=[];vv=[]
    for o in corners:
        loc=grid+o;weights=np.prod(np.where(o,frac,1-frac),axis=1)
        cols=np.fromiter((lookup.get(tuple(c),-1)for c in loc),dtype='i4',count=len(loc));keep=(cols>=0)&(weights>0)
        rr.extend(regions[keep]);cc.extend(cols[keep]);vv.extend(weights[keep])
    B=coo_matrix((vv,(rr,cc)),shape=(len(x),len(control))).tocsr()
    if B[protected].nnz:raise ValueError('Changed a protected contact or bed node')
    live=np.diff(B.indptr)>0;cells=np.flatnonzero(live[T].any(1));tt=T[cells];X=y[tt]
    vol0,shape=geom(y,tt);vr,_=geom(x,tt);D=6*vol0;J=vol0/vr
    grads=gradients(X);u,s,vh=np.linalg.svd(delta/(np.linalg.norm(delta,axis=1)[:,None]+1e-12),full_matrices=False)
    # Choose the axis that can remove the most measured normal penetration.
    axis=vh[0];norm=np.linalg.norm(delta,axis=1);normal=delta/norm[:,None];proj=normal@axis
    scale=.001
    g=coo_matrix(((grads@axis*scale/D[:,None]).ravel(),(np.repeat(np.arange(len(tt)),4),tt.ravel())),shape=(len(tt),len(y))).tocsr()@B
    max_edge=np.max([np.linalg.norm(X[:,a]-X[:,b],axis=1)for a,b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))],axis=0)
    lo=np.maximum(np.minimum(1.,.4/J),np.minimum(1.,1.0003e-5*max_edge**3/D));hi=np.maximum(1.,2.49/J)
    # Required matrix/geometry gates remain active in the native certificate.
    # Use only violated or near-active linear bounds in the small construction solve.
    grid_snap=np.spacing(np.max(abs(y[live]),axis=0).astype('f4')).astype('f8')
    c=len(control);n=len(bad);G=hstack([-B[bad].multiply(proj[:,None]),csr_matrix((n,c)),-eye(n)],format='csr')
    I=eye(c);Z=csr_matrix((c,n));A=vstack([G,hstack([I,-I,Z]),hstack([-I,-I,Z])],format='csr')
    rhs=np.r_[-norm/scale,np.zeros(2*c)];obj=np.r_[np.zeros(c),np.full(c,.02),np.ones(n)]
    bounds=[(-6.,6.)]*c+[(0.,6.)]*c+[(0,None)]*n;steps=[];ret=None;seen=set()
    for j in range(26):
        solve=linprog(obj,A_ub=A,b_ub=rhs,bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-9})
        if not solve.success:steps.append({'iteration':j,'failure':solve.message});break
        w=solve.x[:c];frac_change=g@w;disp=B@w*scale;yy=y.copy();yy[live]=(y[live]+np.rint(disp[live,None]*axis/grid_snap)*grid_snap).astype('f4')
        vol,quality=geom(yy,tt);ratio=vol/vr
        invalid=np.flatnonzero((quality<1e-5)|(vol<=1e-18)|(ratio<.4)|(ratio>2.5))
        steps.append(dict(iteration=j,invalid_cells=len(invalid),sum_remaining_linear_depth_mm=float(solve.fun),max_displacement_mm=float(np.max(abs(disp))*1000)))
        if not len(invalid):
            ret=(yy,axis,w,control,spacing);break
        if j in [0,3,8,16,25] and ret is None:
            for fraction in (.875,.75,.5,.25,.125):
                attempt=y.copy()
                attempt[live]=(y[live]+np.rint(fraction*disp[live,None]*axis/grid_snap)*grid_snap).astype('f4')
                vv,qq=geom(attempt,tt);jj=vv/vr
                if np.all((vv>1e-18)&(qq>=1e-5)&(jj>=.4)&(jj<=2.5)):
                    ret=(attempt,axis,w*fraction,control,spacing)
                    steps[-1]['qualified_fraction']=fraction
                    break
        # Add constraints using the actual edge that sets each element's shape.
        # Bound the edge cubed at the trial, with its gradient, and recheck the emitted shape.
        full=y[tt]+disp[tt,None]*axis
        # Vector argmax over the six actual edges.
        edges=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
        XX=full[invalid];ed=XX[:,edges[:,1]]-XX[:,edges[:,0]];L=np.linalg.norm(ed,axis=2)
        ee=np.argmax(L,axis=1);a,b=edges[ee].T;k=np.arange(len(ee))
        target=np.minimum(shape[invalid],1.0004e-5);vneed=target*L[k,ee]**3/D[invalid]
        dl=3*target*L[k,ee]*(ed[k,ee]@axis)*scale/D[invalid]
        edgeD=(B[tt[invalid,b]]-B[tt[invalid,a]]).multiply(dl[:,None]).tocsr()
        H=edgeD-g[invalid];rk=1+edgeD@w-vneed
        rows=[];limits=[]
        for v_idx,row in enumerate(invalid):
            if row in seen and j>=2:
                # A fixed zero correction is exact at Float32. Pin the local control
                # nodes of a repeatedly unrepresentable cell before rebuilding.
                diff=B[tt[row,1:]]-B[tt[row,0]].tocsr()[[0,0,0]]
                rows.extend([diff,-diff]);limits.extend([np.zeros(3),np.zeros(3)])
            else:
                rows.extend([-g[row],g[row],H[v_idx]]);limits.extend([[1-lo[row]],[hi[row]-1],[rk[v_idx]]])
            seen.add(row)
        extra=vstack(rows,format='csr');A=vstack([A,hstack([extra,csr_matrix((extra.shape[0],c+n))])],format='csr');rhs=np.r_[rhs,np.concatenate(limits)]
    return ret,{'active_cells':len(cells),'controls':c,'steps':steps,'free_nodes':int(live.sum()),'axis':axis.tolist()}

def run(work,lib):
    q=work.parent;j=q.parent.parent;v=q/'qualified-material';x=np.loadtxt(v/'mesh.node',skiprows=1,usecols=(1,2,3));y=np.loadtxt(v/'current.node',skiprows=1,usecols=(1,2,3))
    T=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype='i4')-1;cfg=json.loads((q/'gravity-scene/launch.json').read_bytes());B=banks(Path(cfg['argv'][3]));P=poses(j/'distal-registered-native/target-pose.json');w=np.load(q/'rigid-penetration/enclosed-nodes.npz')
    protected=np.unique(np.r_[np.load(v/'material-walls.npz')['C'].ravel(),np.array(json.loads((q/'gravity-contact/geometry.json').read_bytes())['node_ids'])]);anchors=y[protected]
    results=[];working=y.copy()
    for label,ids in [('r_forearm',[14,16]),('pelvis',[18,19])]:
        nodes=[];normal=[]
        for bank in ids:
            (body,_),(f,p,_) = next(it for it in B.items()if it[0][1]==bank)
            n=w['node'][w['bank']==bank];world=(Rotation.from_quat(P[body,3:]).apply(p)+P[body,:3]).astype('f4').astype('f8')
            nearest_point,distance,_=nearest((world,f),working[n],lib)
            nodes.extend(n.tolist());normal.extend((nearest_point-working[n]).tolist())
        n=np.asarray(nodes);delta=np.array(normal)
        ret,log=fit(working,x,T,n,delta,anchors,protected,.028,lib)
        if ret:
            candidate,axis,coeff,grid,dx=ret
            change=np.flatnonzero(np.any(candidate!=working,axis=1))
            if len(change):
                bits=(work/(label+'-pose.npz'));np.savez_compressed(bits,nodes=change,positions=candidate[change].astype('f4'),direction=axis,grid=grid,spacing=dx,amplitudes_mm=coeff);log.update(delta=digest(bits),moved_nodes=len(change))
                working=candidate
        log['region']=label;results.append(log);print(label,log['steps'][-1],log.get('moved_nodes',0),flush=True)
    ids=np.flatnonzero(np.any(x!=working,axis=1));dest=work/'coupled-registration.npz';np.savez_compressed(dest,nodes=ids,positions=working[ids].astype('f4'))
    record=dict(material=digest(v/'result.json'),source=digest(q/'rigid-penetration/enclosed-nodes.npz'),changes=results,field=digest(dest));(work/'soft-tissue-fit.json').write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':run(Path(__file__).resolve().parent,Path(sys.argv[1]))
