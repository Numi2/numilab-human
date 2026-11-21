"""Compare both Metal paths with exact rational orientations and a triangle oracle."""
from fractions import Fraction
import json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from material_context import sha

def orientation(a,b,c,d):
    x=[p-q for p,q in zip(a,d)];y=[p-q for p,q in zip(b,d)];z=[p-q for p,q in zip(c,d)]
    q=x[0]*(y[1]*z[2]-y[2]*z[1])+x[1]*(y[2]*z[0]-y[0]*z[2])+x[2]*(y[0]*z[1]-y[1]*z[0])
    return int(q>0)-int(q<0)
def triangles(p):
    a,b=p[:3],p[3:];pa=[orientation(*b,v) for v in a];pb=[orientation(*a,v) for v in b]
    if not(min(pa)<0<max(pa) and min(pb)<0<max(pb)):return False
    def edges(source,plane,other):
        for i,x in enumerate(source):
            j=(i+1)%3;y=source[j]
            if plane[i]==plane[j] or plane[i]*plane[j]>0:continue
            signs=[orientation(x,y,other[k],other[(k+1)%3]) for k in range(3)]
            if min(signs)>=0 or max(signs)<=0:return True
        return False
    return edges(a,pa,b) or edges(b,pb,a)
def verify(exe,lib,out):
    rng=np.random.default_rng(66402);n=256
    v=rng.uniform(-.3,.3,(n,6,2,4)).astype('f4')
    v[:,:,1,:]*=np.spacing(np.abs(v[:,:,0,:]));v[:,:,:,3]=0
    plane=v.copy();plane[:,:,0,2]=.125;plane[:,:,1,2]=rng.uniform(-1e-12,1e-12,(n,6)).astype('f4')
    zeros=plane.copy();zeros[:,:,1,2]=0
    tiny=plane.copy();tiny[:,:,1,2]=rng.integers(0,12,(n,6),dtype='u4').view('f4')
    wide=rng.integers(0,2**32,(n,6,2,4),dtype='u4');wide=(wide&0x807fffff)|np.minimum(wide&0x7f800000,np.uint32(0x7f000000))
    inputs=np.vstack((v,plane,zeros,tiny,wide.view('f4')))
    expected=[]
    for row in inputs:
        p=[[Fraction(float(row[k,0,j]))+Fraction(float(row[k,1,j])) for j in range(3)] for k in range(6)]
        sign=orientation(*p[:4]);expected.append([sign,sign,int(triangles(p))])
    with tempfile.TemporaryDirectory() as tmp:
        f=Path(tmp)/'input';d=Path(tmp)/'result';inputs.tofile(f);subprocess.run([exe,lib,f,d],check=True)
        got=np.fromfile(d,'i4').reshape(-1,4)[:,:3];wrong=np.flatnonzero(np.any(got!=expected,axis=1))
        if len(wrong):raise AssertionError([(int(i),got[i].tolist(),expected[i]) for i in wrong[:6]])
    out.write_text(json.dumps({'seed':66402,'orientations':len(inputs),'triangle_pairs':len(inputs),
        'engine_sha256':sha(exe),'test_metallib_sha256':sha(lib),'families':['normal','near coplanar','coplanar','subnormal offsets','full exponent range']},indent=2)+'\n')
if __name__=='__main__':verify(*map(Path,sys.argv[1:]))
