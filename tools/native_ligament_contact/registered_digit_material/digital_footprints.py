"""Infer the dorsal distal-phalanx enthesis from its registered basal rim.

Intersect the bone with a sagittal plane, order the closed section, and measure
surface arclength from the dorsal/proximal support point towards the tuft.
Retain the original triangle and barycentric coordinates at every sample.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from material_context import read_inputs, get_layers, applied, unit
from hand_geometry import Surface

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def contours(surface,base,lateral):
    dist=(surface.v-base)@lateral
    # The one-nanometre generic offset resolves planes coincident with mesh vertices.
    dist=dist+1e-9
    edges=[];locations={};uses={}
    for i,face in enumerate(surface.f):
        crossing=[]
        for a,b in zip(face,np.roll(face,-1)):
            if (dist[a]<0)==(dist[b]<0):continue
            lo,hi=sorted([int(a),int(b)])
            u=dist[lo]/(dist[lo]-dist[hi])
            # Quotient exact source seam positions, not distinct physical boundaries.
            q=surface.v[lo]+u*(surface.v[hi]-surface.v[lo])
            key=tuple(sorted((tuple(surface.v[lo]),tuple(surface.v[hi]))))
            locations[key]=q
            weights=np.zeros(3);weights[np.flatnonzero(face==lo)[0]]=1-u;weights[np.flatnonzero(face==hi)[0]]=u
            crossing.append((key,weights))
        if crossing:
            if len(crossing)!=2:raise ValueError('Nontransverse sagittal cut')
            n=len(edges);edges.append((i,crossing))
            for k,_ in crossing:uses.setdefault(k,[]).append(n)
    if not edges or any(len(x)!=2 for x in uses.values()):raise ValueError('Open sagittal section')
    return edges,locations,uses

def sites(surface,axis,dorsal,lateral,center,offset):
    edges,points,uses=contours(surface,center+offset*lateral,lateral)
    # Axial/dorsal scale is physical metres in both directions. The proximal-
    # dorsal supporting tangent is a reproducible geometric rim proxy.
    rim=max(points,key=lambda k:float((points[k]-center)@(dorsal-axis)))
    def travel(seed):
        chain=[];u=rim;e=seed
        for _ in range(len(edges)):
            face,pts=edges[e];(ki,wi),(kj,wj)=pts
            if kj==u:ki,kj,wi,wj=kj,ki,wj,wi
            if ki!=u:raise ValueError('Broken surface section')
            chain.append((face,ki,kj,wi,wj,np.linalg.norm(points[kj]-points[ki])))
            if kj==rim:break
            nei=uses[kj];e=nei[1] if nei[0]==e else nei[0];u=kj
        return chain
    def sample(chain,s):
        for face,ki,kj,wi,wj,ds in chain:
            if ds>=s:
                t=s/ds
                weights=(1-t)*wi+t*wj
                q=weights@surface.tri[face]
                return q,int(face),weights
            s-=ds
        raise ValueError('Sagittal section too short')
    chains=[travel(seed) for seed in uses[rim]]
    def forward(chain):
        return (sample(chain,.00122)[0]-points[rim])@(axis+dorsal)
    chain=max(chains,key=forward)
    if forward(chain)<=0:raise ValueError('No distal/dorsal tangent from rim')
    return points[rim], [sample(chain,s) for s in (.00101,.00122,.00143)]

def run(w,myoman,c,out):
    cfg=json.loads((w/'current-package.json').read_text());av=cfg['argv'];te=Path(av[av.index('--tendon-payload')+1]);td=json.loads(te.with_suffix('.manifest.json').read_text())
    routes,users,solids,bybody,poses=read_inputs(w/'current-package.json',myoman,c/'emitted-native/stderr.log')
    layer={k:Surface(*v) for k,v in get_layers(w/'qualified/native-coupled-006/native/resting-human.mrvpack',w/'qualified/motion-008/step-8.receipt.json').items()}
    lookup={(r['muscle'],r['endpoint']):r for r in td['endpoints']}
    results=[]
    for name,route in sorted(routes.items()):
        terminal=route[-1];body=terminal['body'];sid,owner=bybody[body][0];surf=Surface(*solids[sid])
        middle=solids[bybody[route[-2]['body']][0][0]][0].mean(0)
        center=surf.v.mean(0);_,_,axes=np.linalg.svd(surf.v-center,full_matrices=False);axis=axes[0]
        if axis@(center-middle)<0:axis=-axis
        flex=lookup[name.replace('EDC','FDP'),'insertion'];assert flex['body_index']==body
        d=terminal['world']-applied(poses[body],flex['resolved_local_point_m']);d=unit(d-axis*(axis@d))
        lat=unit(np.cross(axis,d));z=(surf.v-center)@axis
        lateral=(surf.v-center)@lat
        # Sample three material lines in the central half width. The observed
        # interval is reported as a construction candidate, not a measured footprint.
        width=lateral.max()-lateral.min()
        lines=[]
        for fraction in (-.25,0,.25):
            rim,candidates=sites(surf,axis,d,lat,center,fraction*width)
            found=[]
            for q,face,bary in candidates:
                a,b,c0=surf.tri[face];n=unit(np.cross(b-a,c0-a))
                if n@d<=0:raise ValueError(f'{name}: candidate on non-dorsal face')
                assert np.min(bary)>=-1e-8 and abs(sum(bary)-1)<1e-9
                layers={}
                for k,s in layer.items():
                    hits=s.ray(q+n*1e-8,n,.02)
                    layers[k]=dict(inside=s.inside(q),normal_gap_mm=(1000*(hits[0][0]+1e-8) if hits else None))
                found.append(dict(world_m=q.tolist(),face=face,bary=bary.tolist(),axial_fraction=float(((q-center)@axis-z.min())/np.ptp(z)),normal_dorsal=float(n@d),distance_from_source_mm=float(1000*np.linalg.norm(q-terminal['world'])),layers=layers))
            lines.append(dict(lateral_fraction=fraction,rim_m=rim.tolist(),samples=found))
        results.append(dict(muscle=name,site=terminal['site'],body=body,bone_bank=sid,bone_name=owner,lines=lines))
    out.write_text(json.dumps(dict(bone_sha256=digest(Path(av[3])),myosim_sha256=digest(Path(av[2])),tendon_sha256=digest(te),arclength_mm=[1.01,1.22,1.43],footprints=results),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
