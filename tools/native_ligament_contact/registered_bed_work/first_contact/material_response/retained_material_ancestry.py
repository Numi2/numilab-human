"""Recover ancestry of zero-spatial-overlap recipients through their conserved material tetrahedra."""
import json,sys
from pathlib import Path
import numpy as np
from build_carpal_cavity import FACES
from fit_carpal_clearance import geom
from partition_carpal_material import solid
from audit_registered_skeleton import digest

def run(mesh,overlap,plc,paths,out):
    d=json.loads((overlap/'result.json').read_bytes());ov=np.load(overlap/'overlap.npz');last=np.load(paths[-1]);ids=np.array(d['uncovered_children'],dtype=int);cells=last['t'][ids];roles=last['role'][ids]
    if d['mesh']!=digest(paths[-1]) or d['mapping']!=digest(overlap/'overlap.npz'):raise ValueError('Unmatched spatial audit')
    steps=[];initial=None;previous=None
    for path in paths:
        z=np.load(path);tt=np.sort(z['t'],axis=1);current=z['x'][cells].astype(float)
        for tc,k in zip(cells,roles):
            ii=np.flatnonzero((tt==np.sort(tc)).all(1))
            if len(ii)!=1 or z['role'][ii[0]]!=k:raise ValueError('Material tetrahedron disappeared or changed role')
        v,_=geom(current);minimum=v.min(initial=float('inf'))
        if previous is None:initial=current.copy()
        else:
            # det(A+s B) is cubic. Test both endpoints and all internal stationary points.
            for a,b in zip(previous,current):
                A=a[1:]-a[0];B=(b[1:]-b[0])-A
                def det(M):return float(np.linalg.det(M))/6
                def coeff(mask):return det(np.array([B[i] if mask&(1<<i) else A[i] for i in range(3)]))
                c=np.array([coeff(0),sum(coeff(v) for v in (1,2,4)),sum(coeff(v) for v in (3,5,6)),coeff(7)])
                for s in (0.,1.,*(v.real for v in np.roots([3*c[3],2*c[2],c[1]]) if abs(v.imag)<1e-10 and 0<v.real<1)):
                    minimum=min(minimum,float(np.polynomial.polynomial.polyval(s,c)))
        if minimum<=0:raise ValueError('Nonpositive Jacobian along registered material homotopy')
        steps.append(dict(mesh=digest(path),minimum_volume_m3=float(minimum)));previous=current
    source=np.load(plc)['source_tetrahedra'];data=np.loadtxt(mesh/'mesh.ele',dtype='i4',skiprows=1)[source];xyz=np.loadtxt(mesh/'mesh.node',skiprows=1,usecols=[1,2,3]);a=xyz[data[:,1:5]-1];lower=a.min(1);upper=a.max(1);line=[]
    for c,r,child in zip(initial,roles,ids):
        mid=c.mean(0);s=solid((c-mid)*1000,FACES);v=s.volume();matches=[]
        for i in np.flatnonzero((data[:,5]==r)&((upper>=c.min(0))&(lower<=c.max(0))).all(1)):
            obj=s^solid((a[i]-mid)*1000,FACES)
            if obj.status()!=s.status():raise ValueError('Uncertain material intersection')
            volume=obj.volume()
            if volume>0:matches.append((int(source[i]),float(volume)))
        ratio=sum(vv for _,vv in matches)/v
        if not matches or not .999<ratio<1.001:raise ValueError(f'Original material was uncovered for {child}: {ratio}')
        line.append(dict(child=int(child),role=int(r),native_node_ids=last['t'][child].tolist(),original_volume_mm3=v,covered_fraction=ratio,parents=matches))
    out.mkdir();(out/'result.json').write_text(json.dumps(dict(spatial_audit=digest(overlap/'result.json'),material_tetrahedra=line,homotopy=steps,source_mesh=digest(mesh/'mesh.ele')),indent=2)+'\n')
    print([(k['child'],k['covered_fraction']) for k in line])
if __name__=='__main__':
    mesh,overlap,plc,out,*paths=map(Path,sys.argv[1:]);run(mesh,overlap,plc,paths,out)
