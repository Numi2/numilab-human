"""Measure exposed area samples and protrusion depth on identically tagged muscle surfaces."""
import ctypes as c
import json,runpy,sys
from pathlib import Path
import numpy as np
from registered_forearms import read,sha
from verify_native_extensor_pair import loaded

class Enclosure:
    def __init__(self,lib,points,faces):
        P=np.ctypeslib.ndpointer(dtype='float64',flags='C_CONTIGUOUS');I=np.ctypeslib.ndpointer(dtype='int32',flags='C_CONTIGUOUS')
        l=c.CDLL(str(lib));l.cortex_create.argtypes=[P,c.c_int,I,c.c_int];l.cortex_create.restype=c.c_void_p
        l.skin_enclosure.argtypes=[c.c_void_p,P,c.c_int,I,P];l.skin_enclosure.restype=None
        l.cortex_destroy.argtypes=[c.c_void_p];l.cortex_destroy.restype=None
        tri=points[faces].astype(float);valid=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)>0.
        if not valid.any():raise ValueError('Empty material boundary')
        self.lib=l;self.handle=l.cortex_create(np.ascontiguousarray(points,dtype='f8'),len(points),np.ascontiguousarray(faces[valid],dtype='i4'),int(valid.sum()))
    def close(self):self.lib.cortex_destroy(self.handle)
    def query(self,p):
        p=np.ascontiguousarray(p,dtype='f8');label=np.empty(len(p),'i4');depth=np.empty(len(p),'f8')
        self.lib.skin_enclosure(self.handle,p,len(p),label,depth)
        if not np.isfinite(depth).all():raise ValueError('Nonfinite nearest surface')
        return label,depth

def test(lib):
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    f=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
    s=Enclosure(lib,v,f)
    label,d=s.query([[.1,.1,.1],[1.,1.,1.],[.5,.5,0.]])
    s.close()
    if list(label)!=[1,0,-1] or not np.allclose(d,[.1,2/np.sqrt(3),0]):raise ValueError('Closed tetrahedron test failed')

def measure(s,y,f):
    t=y[f].astype(float);area=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)/2
    # Three-point symmetric quadrature describes extent. Discrete triangle tests
    # independently determine whether a muscle and skin actually cross.
    quadrature=t.mean(1)[:,None,:]*.5 + t*.5
    label,depth=s.query(quadrature.reshape(-1,3));w=np.repeat(area/3,3)
    outside=label==0;unknown=label<0;node,dist=s.query(y)
    return dict(triangles=len(f),surface_area_m2=float(area.sum()),samples=len(label),exterior_samples=int(outside.sum()),
        ambiguous_samples=int(unknown.sum()),sampled_exterior_area_cm2=float(1e4*w[outside].sum()),
        sampled_ambiguous_area_cm2=float(1e4*w[unknown].sum()),exterior_area_fraction=float(w[outside].sum()/area.sum()),
        vertices=len(y),exterior_vertices=int((node==0).sum()),ambiguous_vertices=int((node<0).sum()),
        maximum_exterior_vertex_depth_mm=float(1000*dist[node==0].max(initial=0)),
        volume_integral_depth_mm_m2=float(1000*(w[outside]@depth[outside])))

def audit(case,dense,reader,lib):
    meta,pose,raw=loaded(case,False,dense)
    sys.path.insert(0,str(reader.parent/'src'));pr=runpy.run_path(str(reader))['primitives']
    geo={key:(v,f) for key,_,v,f in pr(raw,pose) if key==(51007,1) or (key[0]==51005 and key[1] in (109,110,137,138))}
    skin=Enclosure(lib,*geo.pop((51007,1)));banks={}
    try:
        for (_,sid),(v,f) in sorted(geo.items()):banks[sid]=measure(skin,v,f)
    finally:skin.close()
    if len(banks)!=4:raise ValueError('Missing bilateral forearm geometry')
    return dict(accepted=meta['accepted_time_s'],pack=meta['pack_file_sha256'],banks=banks)
if __name__=='__main__':
    case,reader,lib=map(Path,sys.argv[1:4]);test(lib.resolve());print(json.dumps(audit(case,'--dense' in sys.argv[4:],reader,lib.resolve()),indent=2))
