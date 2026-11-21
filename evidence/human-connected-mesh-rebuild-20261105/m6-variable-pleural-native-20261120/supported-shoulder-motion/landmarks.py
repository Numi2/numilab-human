"""Locate shoulder reference regions from adjacent named structures in one source atlas."""
import json
from pathlib import Path
import sys
import zipfile
import hashlib
import numpy as np
from scipy.spatial import cKDTree
from compare_registration import prepare
from compare_atlas import vertices_and_faces

def search_regions(launch,anchors,motion,archive):
    _,_,_,an,_,_,data=prepare(launch,anchors,motion,archive)
    labels=(archive.parent/'isa_element_parts.txt').read_text().splitlines()
    parts={r.split('\t')[1]:r.split('\t')[2] for r in labels if len(r.split('\t'))==3}
    outputs=[]
    with zipfile.ZipFile(archive) as z:
        for row in an['candidates']:
            hand='right' if row['side'] else 'left'
            s,h=(data[k] for k in row['stable_ids'])
            htree=cKDTree(h[2]); stree=cKDTree(s[2])
            landmarks={}; roles={}
            for role,label in [('biceps',f'long head of {hand} biceps brachii'),('subscapularis',f'{hand} subscapularis')]:
                name=parts[label]; contents=z.read(next(path for path in z.namelist() if path.endswith('/'+name+'.obj')))
                v,_=vertices_and_faces(contents); roles[role]=v
                landmarks[role]=dict(label=label,member_id=name,member_sha256=hashlib.sha256(contents).hexdigest(),bounds_m=[v.min(axis=0).tolist(),v.max(axis=0).tolist()])
            # The biceps origin is its proximal scapular contact. Search the
            # upper portion along the atlas superior axis and report actual distance.
            b=roles['biceps']; upper=b[:,2]>=np.max(b[:,2])-.012
            dist,ids=stree.query(b[upper]); ix=int(np.argmin(dist)); origin=s[2][ids[ix]]
            sc=roles['subscapularis']; btree=cKDTree(b)
            hs,hi=htree.query(sc); bs,_=btree.query(sc)
            # Find the part of the subscapularis insertion adjacent to the biceps.
            # Limit to the humeral neighborhood before assessing fiber proximity.
            chosen=np.flatnonzero(hs<=.003)
            if len(chosen):j=chosen[np.argmin(bs[chosen])]
            else:j=np.argmin(hs)
            end=h[2][hi[j]]
            landmarks.update(scapular_proximal_biceps_surface_gap_m=float(dist[ix]),
                             humeral_subscapularis_gap_m=float(hs[j]),subscapularis_to_biceps_m=float(bs[j]),
                             candidate_site_separation_m=float(np.linalg.norm(end-origin)),
                             scapular_vertex=int(ids[ix]),humeral_vertex=int(hi[j]),humeral_nearby_sites=len(chosen))
            outputs.append(dict(side=row['side'],atlas=landmarks))
    return outputs

if __name__=='__main__':
    *args,out=map(Path,sys.argv[1:]);out.write_text(json.dumps(search_regions(*args),indent=2)+'\n')
