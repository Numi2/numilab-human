"""Measure triangle-clipped posterior projected area and bed clearance in emitted coordinates."""
import argparse,json,re
from pathlib import Path
import numpy as np
from compile_connected_joint import digest

def clipped_fraction(distance, height):
    # Exact area ratio of a P1 triangle below the cutting plane.
    a,b,c=np.sort(distance,axis=1).T
    f=np.zeros(len(a))
    f[c<=height]=1.
    lo=(a<height)&(height<b)
    f[lo]=(height-a[lo])**2/((b[lo]-a[lo])*(c[lo]-a[lo]))
    hi=(b<=height)&(height<c)
    f[hi]=1.-(c[hi]-height)**2/((c[hi]-a[hi])*(c[hi]-b[hi]))
    return f

def measure(source,bed,output):
    v=source/'volume';g=json.loads(bed.read_bytes());n=np.array(g['normal'])
    y=np.loadtxt(v/'current.node',skiprows=1,usecols=(1,2,3))
    walls=np.load(v/'material-walls.npz');tri=walls['O']
    h=(y[:,0]*n[0]+y[:,1]*n[1]+(y[:,2]-g['height_m'])*n[2])/np.linalg.norm(n)
    tangent=np.cross(y[tri[:,1]]-y[tri[:,0]],y[tri[:,2]]-y[tri[:,0]])
    oriented=tangent @ (n/np.linalg.norm(n))*.5
    anchors=[int(s) for s in re.findall(r'^\{\d+u,(\d+)u,\d+u,\{',(source/'binding/app/NumiHumanLoadedSkinPoseData.inc').read_text(),re.M)]
    if len(anchors)!=14 or np.min(h)<=0:raise ValueError('Unmatched material support')
    present=[]
    for k in ('O','I','D','C'):
        ids=np.unique(walls[k]);present.append(dict(surface=k,min_m=float(min(h[ids])),nodes=len(ids)))
    area_samples=[]
    for height in (.00001,.0001,.00025,.0005,.001,.002,.004,.008,.012,.02):
        f=clipped_fraction(h[tri],height)
        area_samples.append(dict(plane_advance_m=height,posterior_projected_m2=float(f@np.maximum(-oriented,0)),opposite_projected_m2=float(f@np.maximum(oriented,0)),outer_nodes_below=int(np.count_nonzero(h[np.unique(tri)]<height))))
    record=dict(projected_area=area_samples,surface_clearance=present,anchor_min=float(min(h[anchors])),reference=digest(v/'mesh.node'),loaded=digest(v/'current.node'),walls=digest(v/'material-walls.npz'),bed=digest(bed))
    print(record)
    output.write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':
    import sys
    measure(*(Path(a) for a in sys.argv[1:]))
