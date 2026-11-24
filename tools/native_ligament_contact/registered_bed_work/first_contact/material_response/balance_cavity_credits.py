"""Solve an exact local transportation problem on the existing admissible tissue domain."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def rebalance(q,c,w,changed_offset,good,centers,points,region,off,allowed,radius,
              cells,role_ids,components,src,dst):
    keep=np.ones(len(w),bool)
    replacements=[];failed=[];fixed=0
    adjacency=np.argsort(cells.ravel(),kind='stable')
    prefix=np.r_[0,np.cumsum(np.bincount(cells.ravel()))]
    sort=np.argsort(c,kind='stable');edges=np.r_[0,np.cumsum(np.bincount(c,minlength=len(cells)))]
    newbad=np.unique(c[changed_offset:][~good])
    def gather(rows,offsets,index):
        return np.concatenate([index[offsets[k]:offsets[k+1]] for k in rows])
    for _,oldids,n,_,_ in components:
        globaltargets=np.unique(dst[np.isin(src,oldids)])
        if not len(globaltargets) or not np.isin(globaltargets,role_ids).all():
            continue
        targets=np.searchsorted(role_ids,globaltargets)
        if not np.isin(targets,newbad).any():continue
        halo=targets
        for depth in range(3):
            if depth:
                corners=gather(np.unique(cells[halo]),prefix,adjacency)
                halo=np.union1d(halo,corners//4)
            # Unrelated new cells have their own physical cavity and cannot lend capacity here.
            working=halo[(~np.isin(halo,c[changed_offset:])) | np.isin(halo,targets)]
            local=gather(working,edges,sort);local=local[keep[local]]
            # Keep approved exceptional historical edges fixed.
            valid=np.array([region[a] in allowed[off[b]:off[b+1]] for a,b in zip(q[local],c[local])])
            local=local[valid | np.isin(c[local],targets)]
            qids,qi=np.unique(q[local],return_inverse=True);cids,ci=np.unique(c[local],return_inverse=True)
            if not len(qids):continue
            donor=np.bincount(qi,weights=w[local]);capacity=np.bincount(ci,weights=w[local])
            d=np.linalg.norm(points[qids,None,:]-centers[cids][None,:,:],axis=2)
            permissible=d<=radius
            for j,b in enumerate(cids):
                permissible[:,j]&=np.isin(region[qids],allowed[off[b]:off[b+1]])
            if not permissible.any(0).all() or not permissible.any(1).all():continue
            a,b=np.nonzero(permissible);k=np.arange(len(a));N=len(qids)
            A=coo_matrix((np.ones(2*len(k)),(np.r_[a,N+b],np.tile(k,2))),shape=(N+len(cids),len(k))).tocsr()
            # Reward retaining a source-recipient connection already present in the overlay.
            prior=np.isin(a.astype('i8')*len(cids)+b,qi.astype('i8')*len(cids)+ci)
            scale=capacity.min()
            result=linprog((~prior).astype(float)+d[a,b]/radius,A_eq=A,b_eq=np.r_[donor,capacity]/scale,bounds=(0,None),method='highs')
            if not result.success:continue
            value=result.x*scale
            active=value>0
            if max(abs(np.bincount(a,weights=value,minlength=N)-donor))>1e-16 or max(abs(np.bincount(b,weights=value,minlength=len(cids))-capacity))>1e-16:
                continue
            keep[local]=False;replacements.append((qids[a[active]],cids[b[active]],value[active]))
            fixed+=1;break
        else:failed.append(oldids)
    if failed:return None,failed
    combined=[(q[keep],c[keep],w[keep]),*replacements]
    return tuple(np.concatenate([entry[i] for entry in combined]) for i in range(3)),fixed
