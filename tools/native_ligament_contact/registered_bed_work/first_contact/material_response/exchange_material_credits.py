"""Exchange equal source credits across a connected material volume with fixed cell and carrier masses."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

def exchange(q,c,mass,accepted,errors,tets,centers,allowed,max_distance):
    tree=cKDTree(centers);nc=len(centers);inc=coo_matrix((np.ones(tets.size,'i4'),(np.repeat(np.arange(nc),4),tets.ravel())),shape=(nc,int(tets.max())+1)).tocsr()
    by_cell=np.argsort(c,kind='stable');off=np.r_[0,np.cumsum(np.bincount(c,minlength=nc))]
    left=mass.copy();added=[];stats=[];sources=0;max_used=0
    for child in np.unique(c[errors]):
        # Follow continuous material by shared faces. A spatially close neighboring digit or other sheet cannot create a shortcut.
        bank=np.asarray(sorted(tree.query_ball_point(centers[child],max_distance)),dtype='i4')
        corr=inc[bank]@inc[bank].T
        sub=corr.tocoo();i=sub.row[sub.data==3];j=sub.col[sub.data==3]
        distance=np.linalg.norm(centers[bank[i]]-centers[bank[j]],axis=1)
        graph=coo_matrix((distance,(i,j)),shape=(len(bank),len(bank))).tocsr()
        geod=dijkstra(graph,indices=np.searchsorted(bank,child),limit=max_distance)
        inside=bank[geod<np.inf];units=np.concatenate([by_cell[off[v]:off[v+1]] for v in inside]);units=units[accepted[units] & (left[units]>0)]
        viable=allowed(q[units],np.full(len(units),child,dtype='i4'));units=units[viable]
        target_distance=geod[np.searchsorted(bank,c[units])]
        # Shortest swaps first, stable cell/source order breaks ties.
        order=np.lexsort((q[units],c[units],target_distance));units=units[order];target_distance=target_distance[order]
        for source_edge in np.flatnonzero(errors&(c==child)):
            source=q[source_edge];eligible=allowed(np.full(len(units),source,dtype='i4'),c[units])
            for ix in np.flatnonzero(eligible & (left[units]>0)):
                other=units[ix];amount=min(left[other],left[source_edge])
                if amount<=0:continue
                left[other]-=amount;left[source_edge]-=amount;sources+=1
                added.extend(((source,c[other],amount),(q[other],child,amount)))
                max_used=max(max_used,float(target_distance[ix]))
                if left[source_edge]==0:break
        outstanding=float(left[(c==child)&errors].sum())
        if outstanding:stats.append(dict(cell=int(child),kg=outstanding,connected_cells=len(inside)))
    take=left>0;raw=np.asarray(added).reshape(-1,3)
    return (np.r_[q[take],raw[:,0].astype('i4')],np.r_[c[take],raw[:,1].astype('i4')],np.r_[left[take],raw[:,2]],dict(unmatched=stats,balanced_exchanges=sources,maximum_material_path_m=max_used,volume_reach_m=max_distance))

def connected_flow(exe,q,c,w,normal,errors,tets,centers,qpoints,reg,off,rids,radius,allowed):
    from scipy.sparse.csgraph import connected_components
    from solve_connected_carrier_flow import solve
    # Reallocate only existing credits in material connected to the affected receivers.
    tree=cKDTree(centers);nc=len(centers);inc=coo_matrix((np.ones(tets.size,'i4'),(np.repeat(np.arange(nc),4),tets.ravel())),shape=(nc,int(tets.max())+1)).tocsr()
    affected=np.unique(c[errors]);bound=radius
    bank=np.unique(np.concatenate(tree.query_ball_point(centers[affected],bound)))
    corr=(inc[bank]@inc[bank].T).tocoo();i=corr.row[corr.data==3];j=corr.col[corr.data==3];dist=np.linalg.norm(centers[bank[i]]-centers[bank[j]],axis=1)
    graph=coo_matrix((dist,(i,j)),shape=(len(bank),len(bank))).tocsr()
    reach=dijkstra(graph,indices=np.searchsorted(bank,affected),limit=bound,min_only=True)
    all_results=[];history=[]
    for limit in (.020,bound):
        active=reach<=limit;candidate=bank[active];_,label=connected_components(graph[active][:,active],directed=False)
        use=np.zeros(len(w),bool);all_results=[];valid=True
        for comp in np.unique(label[np.searchsorted(candidate,affected)]):
            region_cells=candidate[label==comp]
            use_rows=(normal|errors)&np.isin(c,region_cells);ix=np.flatnonzero(use_rows);qc=q[ix];loc=c[ix];vals=w[ix];dest,cc=np.unique(loc,return_inverse=True);N=len(dest)
            quota=np.bincount(qc,weights=vals,minlength=len(qpoints));recv=np.bincount(cc,weights=vals,minlength=N);allq=np.flatnonzero(quota>0)
            seed=np.column_stack((qc[normal[ix]],loc[normal[ix]],vals[normal[ix]]))
            keys=[qc[normal[ix]].astype('i8')*N+cc[normal[ix]]]
            for rid in np.unique(reg[allq]):
                users=allq[reg[allq]==rid];ok=np.searchsorted(off[1:],np.flatnonzero(rids==rid),side='right');aa=np.flatnonzero(np.isin(dest,ok))
                if not len(aa):continue
                source_pos=qpoints[users];child_pos=centers[dest[aa]]
                # Author both sides, so dilute credits and small receivers each have admissible options.
                for src_tree in [True,False]:
                    src,target=(source_pos,child_pos) if src_tree else (child_pos,source_pos)
                    k=min(8,len(src));distance,neighbor=cKDTree(src).query(target,k=k,distance_upper_bound=radius)
                    distance=distance.reshape(len(target),k);neighbor=neighbor.reshape(len(target),k)
                    jj,kk=np.nonzero(np.isfinite(distance));point=neighbor[jj,kk]
                    qa,ca=(users[point],aa[jj]) if src_tree else (users[jj],aa[point])
                    keys.append(qa.astype('i8')*N+ca)
            v,stats=solve(exe,keys,quota,recv,dest,len(qpoints),seed)
            stats.update(connected_receivers=N,connected_radius_m=limit,active_source_points=len(allq))
            history.append(stats);print('connected exchange',stats,flush=True)
            if stats['child_deficits'] or stats['carrier_deficits']:valid=False;break
            if not np.all(allowed(v[:,0].astype('i4'),v[:,1].astype('i4'))):raise ValueError('Invalid accepted material route')
            use|=use_rows;all_results.append(v)
        if valid:
            a=np.concatenate(all_results);return np.r_[q[~use],a[:,0].astype('i4')],np.r_[c[~use],a[:,1].astype('i4')],np.r_[w[~use],a[:,2]],dict(solves=history,unmatched=[])
    return q,c,w,dict(solves=history,unmatched=history[-1]['unmatched_children'])
