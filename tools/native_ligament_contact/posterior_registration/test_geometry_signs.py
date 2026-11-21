"""Check filtered and exact GPU signs against arbitrary-precision Python integers."""
import json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from material_context import sha
def exact(a):
    pts=[]
    for axis in a.T:
        ratios=[float(v).as_integer_ratio() for v in axis]
        den=max(q for _,q in ratios);pts.append([n*(den//q) for n,q in ratios])
    x,y,z=[[v-t[3] for v in t[:3]] for t in pts]
    v=x[0]*(y[1]*z[2]-y[2]*z[1])+x[1]*(y[2]*z[0]-y[0]*z[2])+x[2]*(y[0]*z[1]-y[1]*z[0])
    return int(v>0)-int(v<0)
def test(exe,lib,out):
    rng=np.random.default_rng(2301);n=1024
    typical=rng.normal(size=(n,6,3)).astype('f4')
    planar=typical.copy();planar[:,:4,2]=np.float32(.0625)
    ulp=planar.copy();ulp[:,3,2]=np.nextafter(ulp[:,3,2],np.where(np.arange(n)%2,-np.inf,np.inf).astype('f4'))
    wide=rng.integers(0,2**32,size=(n,6,3),dtype='u4')
    exp=wide&0x7f800000;wide=(wide&0x807fffff)|np.minimum(exp,np.uint32(0x7f000000))
    w=wide.view('f4');w[:n//3,0,0]=np.array([1],'u4').view('f4')[0]
    a=np.array([[0,0,0],[1,0,0],[0,1,0]],'f4')
    cases={
      'transverse':([[.2,.2,-1],[.2,.2,1],[.8,.8,1]],True),
      'separated_straddling_infinite_planes':([[2,2,-1],[2,2,1],[8,8,1]],False),
      'coplanar':([[0,0,0],[1,0,0],[0,1,0]],False),
      'one_sided_touch':([[.2,.2,0],[.2,.8,1],[.4,.4,1]],False),
      'vertex_on_plane_crossing':([[.2,.2,0],[.4,.2,-1],[.3,.6,1]],True),
      'edge_only':([[0,0,0],[1,0,0],[0,1,1]],False)}
    fixtures=[];wanted=[]
    for tri,answer in cases.values():
        for f,g in ((a,tri),(tri,a)):
            for shift in range(3):
                fixtures.append(np.vstack((np.roll(f,shift,axis=0),np.flip(g,axis=0))))
                wanted.append(answer)
    p=np.vstack((typical,planar,ulp,w,np.array(fixtures,'f4')))
    padded=np.zeros((len(p),6,4),'f4');padded[:,:,:3]=p
    with tempfile.TemporaryDirectory() as tmp:
        src=Path(tmp)/'fixtures';res=Path(tmp)/'results';padded.tofile(src)
        subprocess.run([exe,lib,src,res],check=True);v=np.fromfile(res,'i4').reshape(-1,4)
    target=np.array([exact(q[:4]) for q in p])
    if not np.array_equal(v[:,0],target) or not np.array_equal(v[:,1],target):
        raise ValueError(('wrong determinant sign',np.flatnonzero((v[:,0]!=target)|(v[:,1]!=target))[:10].tolist()))
    if not np.array_equal(v[-len(wanted):,2],wanted):raise ValueError('triangle crossing fixture mismatch')
    out.write_text(json.dumps(dict(seed=2301,orientation_cases=len(target),tested_filtered_and_exact=True,triangle_cases=len(fixtures),cases=list(cases),native_library=sha(lib),native_test=sha(exe)),indent=2)+'\n')
if __name__=='__main__':test(*map(Path,sys.argv[1:]))
