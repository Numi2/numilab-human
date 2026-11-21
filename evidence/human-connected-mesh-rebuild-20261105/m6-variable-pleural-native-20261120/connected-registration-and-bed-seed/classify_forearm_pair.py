"""Classify exact foreground witnesses between bilateral BRD and FPL surface banks."""
from collections import Counter
import json

import numpy as np

from fit_spline import nearby_pairs
from native_source import banks

for left,right in ((109,135),(110,136)):
    key = (51005,right)
    xr,fr = banks[(51005,left)]
    xt,ft = banks[key]
    hits = np.asarray(sorted(nearby_pairs(xr,fr,{key:(xt,ft)}).get(key,set())),dtype=int)
    assert hits.shape[1] == 2
    a,b = xr[fr[hits[:,0]]].astype(float),xt[ft[hits[:,1]]].astype(float)
    na = np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0])
    nb = np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
    na /= np.linalg.norm(na,axis=1)[:,None]
    nb /= np.linalg.norm(nb,axis=1)[:,None]
    d1 = np.einsum('nki,ni->nk',b-a[:,:1],na)
    d2 = np.einsum('nki,ni->nk',a-b[:,:1],nb)
    common = np.all(a[:,:,None] == b[:,None,:],axis=-1).sum(axis=(1,2))
    cosine = np.sum(na*nb,axis=1)
    plane_extent = np.maximum(abs(d1).max(axis=1),abs(d2).max(axis=1))
    # Exact Float32 edge/triangle coincidence is reported separately from
    # geometric plane proximity. No source-contact exemptions are inferred.
    prox = {}
    for tolerance in (0.,1e-7,5e-7,1e-6,1e-5):
        rows = plane_extent<=tolerance
        prox[str(tolerance)] = dict(pairs=int(rows.sum()),
            opposing_normal_pairs=int(np.count_nonzero(rows & (cosine<-.99))),
            aligned_normal_pairs=int(np.count_nonzero(rows & (cosine>.99))))
    strictly_transverse = (d1.min(axis=1)<0)&(d1.max(axis=1)>0) & (d2.min(axis=1)<0)&(d2.max(axis=1)>0)
    robust_transverse = (d1.min(axis=1)<-1e-6)&(d1.max(axis=1)>1e-6) & (d2.min(axis=1)<-1e-6)&(d2.max(axis=1)>1e-6)
    print(json.dumps(dict(banks=[left,right],raw_pairs=len(hits),
        exactly_shared_coordinates=dict(Counter(map(int,common))),
        conditional_plane_proximity_m=prox,
        strict_transverse_pairs=int(strictly_transverse.sum()),
        at_least_1um_transverse_pairs=int(robust_transverse.sum()),
        maximum_plane_extent_m=float(plane_extent.max()),
        deepest_witness=hits[np.argmax(plane_extent)].tolist())),flush=True)
