"""Find a source-sided surface separation while constraining the actual incident material volume."""
import ctypes,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,csr_matrix,eye,hstack,vstack
from certify_transported_material import measure
from compile_connected_joint import digest

def star_planes(source,faces,pairs):
    edges=[];normals=[];margins=[]
    for a,b in pairs:
        at,bt=faces[a],faces[b];common=set(at)&set(bt)
        ab=np.array([(i,j) for i in at for j in bt if i!=j and not(i in common and j in common)])
        dr=(source[ab[:,1]]-source[ab[:,0]])*1e6
        eq=None
        if len(common)>1:
            shared=source[sorted(common)];eq=np.column_stack(((shared[1:]-shared[:1])*1e6,np.zeros(len(shared)-1)))
        res=linprog([0,0,0,-1],A_ub=np.column_stack((-dr,np.ones(len(dr)))),b_ub=np.zeros(len(dr)),A_eq=eq,b_eq=None if eq is None else np.zeros(len(eq)),bounds=[(-1,1)]*3+[(None,None)])
        if not res.success or res.x[3]<-1e-6:raise ValueError('Source does not admit separating plane')
        n=res.x[:3]/np.linalg.norm(res.x[:3]);gap=float(np.min(dr@n));dist=min(.6,max(gap,0)*.3)
        for i,j in ab:edges.append((i,j));normals.append(n);margins.append(dist)
        if len(common)>1:
            for i in sorted(common)[1:]:
                j=min(common);edges.extend([(i,j),(j,i)]);normals.extend([n,n]);margins.extend([-1e-6,-1e-6])
    return np.array(edges),np.array(normals),np.array(margins)

def run(job,h,library,out,with_density=True):
    z=np.load(job/'registered-boundary/registered-boundary.npz')
    w=np.array(json.loads((job/'connected-layer-crossings/reference.json').read_bytes())['witnesses']['2-2'],dtype='u4')
    f=np.ascontiguousarray(z['D_node_joint'][z['D_face_nodes']],dtype='u4')
    domain=h/'closed-native/native'
    x=np.loadtxt(domain/'mesh.node',skiprows=1,usecols=(1,2,3)).astype('f4').astype(float)
    mesh=np.loadtxt(domain/'mesh.ele',skiprows=1,dtype='i4');e=mesh[:,1:5]-1;mat=mesh[:,5]
    c=np.load(h/'needle-conditioned/domain.npz');mapper=np.full(len(c['x']),-1,dtype='i4');begin=int(np.load(domain/'material-source-indices.npz')['child_begin']);mapper[c['t']]=e[begin:]
    assert np.array_equal(mapper[c['t']],e[begin:])
    src=x.copy();saved=np.load(job/'deep-fat-source-replay/source-displacements.npz');src[saved['nodes']]+=saved['delta']
    ab,n,margin=star_planes(src,f,w);surface_nodes=np.unique(f[np.unique(w)])
    ring=np.unique(e[np.isin(e,surface_nodes).any(1)])
    pinned=np.unique(np.r_[mapper[c['locked']],z['O_node_joint'],z['I_node_joint']]);v=np.setdiff1d(ring,pinned)
    # Restrict all motion to the material assembly for which cortical constraints are available.
    v=np.intersect1d(v,mapper[mapper>=0]);ix=np.full(len(x),-1,'i4');ix[v]=np.arange(len(v))
    tt=np.flatnonzero(np.isin(e,v).any(1));t=e[tt];lookup=ix[t];u=len(v)*3
    ir,ic=np.where(lookup>=0);cols=(3*lookup[ir,ic,None]+np.arange(3)).ravel()
    def cell_matrix(g,scale):
        data=(g[ir,ic]/scale[ir,None]*1e-6).ravel()
        return coo_matrix((data,(np.repeat(ir,3),cols)),shape=(len(t),u)).tocsr()
    def side_matrix(two,norm):
        row=[];col=[];coef=[]
        for k,(i,j) in enumerate(two):
            for item,sign in ((i,-1),(j,1)):
                if ix[item]>=0:
                    row.extend([k]*3);col.extend(3*ix[item]+np.arange(3));coef.extend(sign*norm[k])
        return coo_matrix((coef,(row,col)),shape=(len(two),u)).tocsr()
    G=side_matrix(ab,n);p=np.load(h.parent/'cortical-planes.npz');body=mapper[p['node']];sel=(body>=0)&(ix[body]>=0)
    body=body[sel];norm=p['normal'][sel];on=p['on_plane'][sel];req=np.minimum(np.sum(norm*(x[body]-on),axis=1)*1e6,.08)
    C=coo_matrix((norm.ravel(),(np.repeat(np.arange(len(body)),3),(3*ix[body,None]+np.arange(3)).ravel())),shape=(len(body),u)).tocsr()
    mass=np.fromfile(domain/'cell-mass.f64','<f8')[tt];lo=6*mass/np.where(mat[tt]==15,1100.,950.);hi=6*mass/np.where(mat[tt]==15,900.,850.)
    d0,l0=measure(x,t)
    if np.any(d0<=0):raise ValueError('Invalid baseline')
    y=x.copy();arg=np.ctypeslib.ndpointer(flags='C_CONTIGUOUS');fn=ctypes.CDLL(str(library)).batch;fn.argtypes=[arg,arg,arg,ctypes.c_uint,arg];fn.restype=None
    out.mkdir();history=[];edges=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    def metrics(Y):
        d,l=measure(Y,t);s=margin-np.sum(n*(Y[ab[:,1]]-Y[ab[:,0]]),axis=1)*1e6
        c=req-np.sum(norm*(Y[body]-on),axis=1)*1e6
        # Keep shape tolerance exact independently of the optimizer's linear model.
        m=np.maximum(6e-18,l*1e-5)
        hit=np.zeros(len(w),'u1');fn(Y,f,w,len(hit),hit)
        dens=np.maximum(np.maximum(lo-d,d-hi)/d0,0);fail=np.maximum((m-d)/d0,0)
        return dict(crossings=int(hit.sum()),inverted=int((d<=0).sum()),quality=int((d<m).sum()),density=int((dens>1e-10).sum()),cortex_um=float(max(0,c.max(initial=0))),separation_um=float(max(s.max(),0)),max_density_error=float(dens.max()),max_shape_error=float(fail.max())),(s,c,d,l)
    for it in range(30):
        stats,(s,g,d,l)=metrics(y);history.append(stats);print(it,stats,flush=True)
        if stats['crossings']==stats['quality']==0 and (not with_density or stats['density']==0) and stats['cortex_um']<.002:break
        points=y[t];a,b,c0=(points[:,k]-points[:,0] for k in [1,2,3]);dg=np.stack((np.cross(b,c0),np.cross(c0,a),np.cross(a,b)),1);dg=np.concatenate((-dg.sum(1)[:,None],dg),axis=1)
        D=cell_matrix(dg,d0);de=points[:,edges[:,0]]-points[:,edges[:,1]];el=(de*de).sum(2);ee=np.argmax(el,axis=1);idx=np.arange(len(t));el=np.sqrt(el[idx,ee]);guard=np.maximum(6e-18,1e-5*l)
        active=guard>6e-18;der=dg.copy();adj=(3e-5*el*active)[:,None]*de[idx,ee];der[idx,edges[ee,0]]-=adj;der[idx,edges[ee,1]]+=adj
        Q=cell_matrix(der,d0)
        # Allow the identity iterate on a bound while targeting a small safety margin when a cell can move.
        margin_q=np.minimum(np.maximum(d0-np.maximum(6e-18,l0*1e-5),0)*.2,d0*.0003)
        hard=[-Q,-C];rhs=[(d-guard-margin_q)/d0,-g+.00008]
        if with_density:
            hard.extend([-D,D]);rhs.extend([(d-lo)/d0+2e-11,(hi-d)/d0+2e-11])
        M=vstack(hard).tocsr();rhs=np.concatenate(rhs)
        # Joint elastic phase followed by a minimum-motion phase. Use normalized
        # material deficit separately from separation deficit measured in micrometres.
        hard_k=500.;ng=len(s)
        H=vstack([hstack([hstack([M,csr_matrix((len(rhs),1))]),-np.ones((len(rhs),1))/hard_k]),hstack([-G,-np.ones((ng,1)),csr_matrix((ng,1))])]).tocsr()
        R=np.r_[rhs,-s];score=lambda state: state['separation_um']+hard_k*max(state['max_shape_error'],state['max_density_error'] if with_density else 0.,state['cortex_um'])
        best=None;start=score(stats)
        for trust in (1.,4.,12.,35.):
            bounds=[(-trust,trust)]*u+[(0,None),(0,None)]
            r=linprog(np.r_[np.zeros(u),1,1],A_ub=H,b_ub=R,bounds=bounds,method='highs')
            print('  trust',trust,r.status,r.message if not r.success else r.fun,flush=True)
            if not r.success:continue
            I=eye(u,format='csr');H2=vstack([hstack([H,csr_matrix((H.shape[0],u))]),hstack([I,csr_matrix((u,2)),-I]),hstack([-I,csr_matrix((u,2)),-I]),csr_matrix(np.r_[np.zeros(u),1.,1.,np.zeros(u)][None,:])]).tocsr()
            rr=linprog(np.r_[np.zeros(u+2),np.ones(u)],A_ub=H2,b_ub=np.r_[R,np.zeros(2*u),r.fun+1e-5],bounds=bounds+[(0,None)]*u)
            step=(rr if rr.success else r).x[:u].reshape(-1,3)
            for alpha in (1.,.75,.50,.25,.125):
                new=y.copy();new[v]=(y[v]+alpha*1e-6*step).astype('f4');ss,_=metrics(new);value=score(ss)
                if alpha==.5:print('  tested half',ss,flush=True)
                if value<start-1e-7 and (best is None or value<best[0]):best=(value,new)
        if best is None:break
        y=best[1]
    last,_=metrics(y);ids=np.flatnonzero(np.any(x!=y,axis=1));np.savez_compressed(out/'candidate.npz',nodes=ids,positions=y[ids].astype('f4'))
    (out/'result.json').write_text(json.dumps(dict(with_density=with_density,stages=history,final=last,moved=len(ids),max_displacement_um=float(np.linalg.norm(x[ids]-y[ids],axis=1).max(initial=0)*1e6),affected_cells=len(t),candidate=digest(out/'candidate.npz')),indent=2)+'\n')
if __name__=='__main__':run(*(Path(a).resolve() for a in sys.argv[1:5]),with_density=sys.argv[5]=='density')
