"""Locate detached component nodes and cell centers in the main registered material."""
from functools import lru_cache
import numpy as np
from scipy.spatial import cKDTree


def sub(a,b):
    return tuple(v-w for v,w in zip(a,b))


def det(a,b,c):
    return (a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]))


def contains(corners,point):
    a,b,c,d=corners
    e,f,g=sub(b,a),sub(c,a),sub(d,a)
    orientation=det(e,f,g)
    if orientation<=0:
        raise ValueError('Invalid host cell')
    p=sub(point,a)
    s,t,u=det(p,f,g),det(e,p,g),det(e,f,p)
    values=(orientation-s-t-u,s,t,u)
    return 2 if min(values)>0 else int(min(values)==0)


def locate(points,tets,tags,labels,main,fold):
    minor=np.flatnonzero(labels!=main)
    others=tets[labels[tets[:,0]]!=main]
    which=np.r_[labels[minor],labels[others[:,0]]]
    # An exact common integer lattice represents every finite binary32 value.
    @lru_cache(maxsize=None)
    def integer(node):
        return tuple(int(np.ldexp(float(v),149)) for v in points[node])
    queries=[tuple(4*v for v in integer(n)) for n in minor]
    for tet in others:
        queries.append(tuple(sum(v) for v in zip(*(integer(n) for n in tet))))
    coordinates=np.array([[np.ldexp(float(s),-151) for s in q] for q in queries])
    if not len(queries):
        return dict(exact_candidate_tests=0,components_with_all_vertices_in_main_fat=0)
    tree=cKDTree(coordinates)
    membership=np.zeros((len(queries),2),dtype='u1')
    hosts=np.flatnonzero(labels[tets[:,0]]==main)
    domainlo=coordinates.min(0)
    domainhi=coordinates.max(0)
    checks=0
    for group in np.array_split(hosts,max(1,len(hosts)//8192)):
        corners=points[tets[group]].astype(float)
        lo,hi=corners.min(1),corners.max(1)
        use=np.flatnonzero(np.all(hi>=domainlo,1)&np.all(lo<=domainhi,1))
        if not len(use):
            continue
        center=(lo[use]+hi[use])*.5
        # Inclusive L-infinity cubes enclose the exact Float32 AABBs.
        radius=np.nextafter(np.max(np.maximum(hi[use]-center,center-lo[use]),axis=1),np.inf)
        possible=tree.query_ball_point(center,radius,p=np.inf)
        for slot,candidates in zip(use,possible):
            if not candidates:
                continue
            idx=np.asarray(candidates)
            idx=idx[np.all(coordinates[idx]>=lo[slot],axis=1)&np.all(coordinates[idx]<=hi[slot],axis=1)]
            if not len(idx):
                continue
            row=group[slot]
            h=[tuple(4*v for v in integer(n)) for n in tets[row]]
            role=tags[row]-15
            for i in idx:
                if membership[i,role]==2:
                    continue
                checks+=1
                membership[i,role]=max(membership[i,role],contains(h,queries[i]))
    def summarize(sel):
        a=membership[sel]
        return dict(samples=len(a),strictly_inside_main_fat=int((a[:,1]==2).sum()),
                    on_main_fat_boundary_only=int((a[:,1]==1).sum()),strictly_inside_main_dermis=int((a[:,0]==2).sum()),
                    outside_every_main_tet=int((a.max(1)==0).sum()))
    is_vertex=np.arange(len(queries))<len(minor)
    components=np.unique(which)
    return dict(exact_candidate_tests=checks,registered_vertices=summarize(is_vertex),cell_barycenters=summarize(~is_vertex),
                fold_vertices=summarize(is_vertex&(which==fold)),fold_centers=summarize(~is_vertex&(which==fold)),
                components_with_all_vertices_in_main_fat=sum(int(np.all(membership[is_vertex&(which==c),1]>0)) for c in components),
                scope='Exact point-in-tetrahedron tests on the loaded Float32 mesh. Sampling vertices and centers allows positive overlap witnesses; it does not certify entire cell containment.')
