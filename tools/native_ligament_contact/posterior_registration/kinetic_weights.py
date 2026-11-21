"""Reconstruct a selected cortical vertex's current orthogonal material-face projection."""
import numpy as np

def contact_weights(index,pool,features,threshold,points,rotation,position):
    row=pool[index]
    if index<threshold:return row[6].astype(float)
    weights=features[index-threshold]
    nodes=row[5].copy().view('u4');v=weights@points[nodes];owner=int(row[3].copy().view('u4')[2])
    bone=rotation[128].inv().apply(rotation[owner].apply(row[2,:3])+position[owner][:3])
    e=v[1]-v[0];f=v[2]-v[0];n=np.cross(e,f);k=n@n;rel=bone-v[0]
    a=np.cross(rel,f)@n/k;b=np.cross(e,rel)@n/k
    if min(a,b,1-a-b)<-1e-4:raise ValueError(('Active reverse contact outside material face',index,a,b))
    return np.array([1-a-b,a,b])@weights
