"""Exact coordinate welding and parity for the two emitted closed surfaces."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from count_accepted_cortical_crossings import dyadic,plane
from compare_hand_interfaces import query

def prepare(vertices,faces,predicate):
    vertices,inv=np.unique(vertices,axis=0,return_inverse=True)
    faces=inv[faces]
    used=np.unique(faces)
    if len(used)!=len(vertices):raise ValueError('Unused native surface vertices')
    edges=faces[:,[0,1,1,2,2,0]].reshape(-1,2)
    _,group,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    orientation=np.bincount(group,weights=np.where(edges[:,0]<edges[:,1],1,-1))
    graph=coo_matrix((np.ones(len(edges)),edges.T),shape=(len(vertices),len(vertices))).tocsr()
    comps,labels=connected_components(graph,directed=False)
    exact=dyadic(vertices)
    degenerate=sum(not any(plane(*(exact[k] for k in tri))) for tri in faces)
    hits=query(predicate,vertices,faces,np.ones(len(faces),dtype=bool))
    separate=[(i,j) for i,j in hits if not set(faces[i]).intersection(faces[j])]
    status=dict(components=int(comps),open_or_nonmanifold_edges=int(np.count_nonzero(counts!=2)),
        inconsistent_edges=int(np.count_nonzero(orientation)),degenerate_faces=degenerate,
        nonadjacent_self_intersections=len(separate))
    return vertices,faces,labels,status

def classify(point,point_exact,target,triangles,target_exact):
    # A ray through a vertex or along an edge is ambiguous. Use another axis
    # or a different vertex of this separated component in that case.
    for dim in range(3):
        y,z=(dim+1)%3,(dim+2)%3
        xyz=target[triangles]
        keep=(xyz[:,:,dim].max(axis=1)>=point[dim])
        for a in (y,z):keep&=(xyz[:,:,a].min(axis=1)<=point[a])&(xyz[:,:,a].max(axis=1)>=point[a])
        hits=0
        for t in triangles[keep]:
            a,b,c=(target_exact[k] for k in t)
            def orient(v,w,p):
                return (w[y]-v[y])*(p[z]-v[z])-(w[z]-v[z])*(p[y]-v[y])
            det=orient(a,b,c)
            w1,w2,w3=orient(b,c,point_exact),orient(c,a,point_exact),orient(a,b,point_exact)
            if det==0:
                if w1==w2==w3==0:break
                continue
            s=1 if det>0 else -1
            if min(s*w1,s*w2,s*w3)<0:continue
            d=s*(w1*(a[dim]-point_exact[dim])+w2*(b[dim]-point_exact[dim])+w3*(c[dim]-point_exact[dim]))
            if d<0:continue
            if d==0:return 'boundary'
            if w1*w2*w3==0:break
            hits+=1
        else:return 'inside' if hits%2 else 'outside'
    return None

def compare(a,b):
    va,fa,la,sa=a;vb,fb,lb,sb=b
    values=dyadic(np.vstack((va,vb)));xa,xb=values[:len(va)],values[len(va):]
    sides={}
    for key,vertices,labels,x,other,faces,ox in (
        ('material_in_cortex',va,la,xa,vb,fb,xb),('cortex_in_material',vb,lb,xb,va,fa,xa)):
        results=[]
        for comp in range(int(labels.max())+1):
            for i in np.flatnonzero(labels==comp):
                inside=classify(vertices[i],x[i],other,faces,ox)
                if inside is not None:
                    results.append({'component':comp,'vertex':int(i),'location':inside})
                    break
            else:raise ValueError('Unresolved component ray')
        sides[key]=results
    ok=all(val==0 for s in (sa,sb) for k,val in s.items() if k!='components')
    outside=all(row['location']=='outside' for rows in sides.values() for row in rows)
    return {'material_topology':sa,'cortical_topology':sb,'component_tests':sides,
            'closed_and_exterior':bool(ok and outside)}
