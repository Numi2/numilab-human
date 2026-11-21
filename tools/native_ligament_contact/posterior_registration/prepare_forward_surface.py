"""Bind the forward distance queries to the closed, emitted P1 surface."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial import cKDTree
from fit_contact_domain import points
from material_context import sha
def prepare(part,prev,base,out):
    spec=json.loads((base/'manifest.json').read_text());g=spec['closed_reverse']
    pool=np.fromfile(base/'pool-material-points.bin','f4').reshape(-1,7,4)
    orig=np.fromfile(base/'cortical-triangles.bin','f4').reshape(-1,10,4)
    cfaces=np.fromfile(base/'cortical-certificate-faces.bin','u4').reshape(-1,4)
    idx=np.full((len(orig),4),2**32-1,'u4');cf=pool[g['first']:g['first']+g['unique_vertices']];ids=cf[:,3].copy().view('u4')
    for bank in spec['cortical_certificate']['banks']:
        side=bank['muscle']-29;body=bank['cortical_body'];donor=(131,145)[side]
        ix=np.flatnonzero((ids[:,1]==donor)&(ids[:,2]==body));sp=np.unique(cf[ix,4,:2].copy().view('u4'),axis=0)
        start,n=map(int,sp[0]);assert len(sp)==1 and n==bank['count']
        block=cfaces[bank['first']:bank['first']+n].copy()
        block[:,3]=(128,1,131,145).index(body)
        if not np.array_equal(cf[block[:,:3],2,:3],orig[start:start+n,:3,:3]):raise ValueError('cortical frame mismatch')
        idx[start:start+n]=block
    if np.any(idx==2**32-1):raise ValueError('uncertified cortical triangle')
    rows=np.zeros((g['first'],2,4),'f4');q=pool[:g['first']];side_ids=q[:,3,1].copy().view('u4')
    reports=[]
    for side,sid in enumerate((29,30)):
        s=np.load(part/'conforming-boundary'/f'{sid}.npz');v=s['vertices'];f=s['face'];tree=cKDTree(v);bycell={}
        for j,c in enumerate(s['owner']):bycell.setdefault(c,[]).append(j)
        p=np.load(part/'projected'/f'{sid}.npz');material=p['vertices'].astype(float);node,w,*_=points(part,prev,sid)
        keys={(r.tobytes(),wt.tobytes()):j for j,(r,wt) in enumerate(zip(node.astype('u4'),w.astype('f4')))}
        cellid={tuple(row):j for j,row in enumerate(p['cells'])};regs=np.flatnonzero(side_ids==(131,145)[side])
        local=q[regs,5].copy().view('u4')-spec['contact_flexibility'][side]['first_node']
        refs=np.array([keys[(a.tobytes(),b.tobytes())] for a,b in zip(local,q[regs,6])]);x=np.einsum('ij,ijk->ik',w[refs],material[local])
        near,iv=tree.query(x);tri=np.repeat(iv[:,None],3,axis=1);bary=np.zeros((len(x),3));bary[:,0]=1
        for j in np.flatnonzero(near>2.e-12):
            candidates=np.array(bycell[cellid[tuple(local[j])]],dtype=int)
            y=v[f[candidates]];e=y[:,1]-y[:,0];h=y[:,2]-y[:,0];n=np.cross(e,h);rr=x[j]-y[:,0]
            scale=np.einsum('ij,ij->i',n,n)
            u=np.einsum('ij,ij->i',np.cross(rr,h),n)/scale;z=np.einsum('ij,ij->i',np.cross(e,rr),n)/scale
            ws=np.stack((1-u-z,u,z),axis=1)
            err=abs(np.einsum('ij,ij->i',rr,n))/np.sqrt(scale)
            inside=np.flatnonzero((ws.min(1)>=-2.e-8)&(err<2.e-12))
            if not len(inside):raise ValueError(('no physical face at forward sample',sid,int(regs[j]),min(err),ws.min(1).max()))
            k=min(inside,key=lambda k:(err[k],int(candidates[k])));tri[j]=f[candidates[k]];bary[j]=ws[k]
        residual=np.linalg.norm(np.einsum('ij,ijk->ik',bary,v[tri])-x,axis=1)
        if residual.max()>2.e-12:raise ValueError(('false surface binding',sid,residual.max()))
        global_ids=tri.astype('u4')+np.uint32(g['sides'][side]['first_vertex'])
        rows.view('u4')[regs,0,:3]=global_ids
        rows[regs,1,:3]=bary
        if not np.array_equal(rows.view('u4')[regs,0,:3],global_ids):raise ValueError('Lost emitted surface indices')
        pts=v[tri];matched=pts[:,0]+rows[regs,1,1:2]*(pts[:,1]-pts[:,0])+rows[regs,1,2:3]*(pts[:,2]-pts[:,0])
        emitted_error=float(np.linalg.norm(matched-x,axis=1).max())
        if emitted_error>5e-9:raise ValueError(('rounded sample drift',sid,emitted_error))
        reports.append(dict(muscle=sid,points=len(regs),maximum_packed_rest_error_m=emitted_error,vertex_sites=int(sum(near<=2.e-12)),max_interpolation_error_m=float(residual.max())))
    out.mkdir(exist_ok=False)
    for p in base.iterdir():
        if p.name!='manifest.json':(out/p.name).hardlink_to(p)
    for n,m in [('forward-surface-samples',rows),('forward-cortical-faces',idx)]:
        p=out/(n+'.bin');m.tofile(p);spec['files'][p.name]=sha(p)
    spec['shared_forward_surface']={'parent':sha(base/'manifest.json'),'regions':reports}
    (out/'manifest.json').write_text(json.dumps(spec,indent=2)+'\n');print(reports)
if __name__=='__main__':prepare(*map(Path,sys.argv[1:]))
