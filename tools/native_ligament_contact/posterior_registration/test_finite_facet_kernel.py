"""Compare the actual clipped-contact kernel to independent convex-intersection witnesses."""
import json,sys,subprocess,tempfile
from io import StringIO
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from material_context import sha

def intersect(uv):
    # Intersection of six half-planes in material barycentric coordinates.
    b=np.column_stack((1-uv.sum(1),uv));m=np.vstack((np.eye(3),b.T));c=m[:,0];a=m[:,1:]-c[:,None];pts=[]
    for i in range(len(a)):
        for j in range(i):
            sub=a[[i,j]]
            if abs(np.linalg.det(sub))<1e-13:continue
            xy=np.linalg.solve(sub,-c[[i,j]])
            if min(a@xy+c)>-1e-10:pts.append(xy)
    if len(pts)<3:return None
    pts=np.unique(np.round(pts,11),axis=0)
    if len(pts)<3 or np.linalg.matrix_rank(pts-pts[0])<2:return None
    v=pts[ConvexHull(pts).vertices];q=np.roll(v,-1,axis=0);w=v[:,0]*q[:,1]-v[:,1]*q[:,0];center=np.sum((v+q)*w[:,None],axis=0)/(3*w.sum())
    return np.r_[1-center.sum(),center],np.column_stack((1-v.sum(1),v))

def run(engine,lib,assets,out):
    d=json.loads((assets/'manifest.json').read_text());first=d['finite_facets']['start']
    cases={'contained':[[.1,.1],[.1,.35],[.4,.1]],'three_edge_clip':[[-.5,.3],[.3,1.6],[1.4,-.5]],
        'sliding_off':[[.58,.42],[.69,.45],[.69,.36]],'disjoint':[[.8,.9],[.8,1.2],[1.3,.9]],
        'common_edge':[[0,0],[-.4,.5],[0,.9]],'moving_cortex':[[.1,.1],[.1,.35],[.4,.1]]}
    n=len(cases);p=np.zeros((n,17,4),'f4');t=np.zeros((n,10,4),'f4');delta=np.zeros((4*n,4),'f4');poses=np.zeros((146,2,4),'f4');poses[:,1,3]=1;low=np.zeros((146,4),'f4');ref=np.zeros((8,4),'f4');ref[1::2,3]=1
    poses[1,0,:3]=[.0005,-.001,.0002];L=.02;heights=np.array([-.0002,.00035,-.00014],dtype='f4')
    expect=[]
    for i,(name,points) in enumerate(cases.items()):
        uv=np.asarray(points,'f4');bone=1 if name=='moving_cortex' else 128
        t[i,:3,:3]=[[0,0,0],[L,0,0],[0,L,0]];t[i,3,2]=1
        p[i,0].view('u4')[:]=[i,bone,0,i];p[i,1:4,:3]=np.eye(3);p[i,4:7,:2]=uv;p[i,7:10,:2]=uv*L;p[i,10:13,:2]=uv*L
        p[i,16].view('u4')[:]=np.arange(i*4,i*4+4);delta[4*i:4*i+3,2]=heights
        shift=poses[bone,0,:3].astype(float);expect.append((intersect(uv.astype(float)-shift[:2]/L),heights.astype(float)-shift[2]))
    with tempfile.TemporaryDirectory() as folder:
        fixture=Path(folder)/'fixture'
        with fixture.open('wb') as f:
            for a in [np.array([n,first],'u4'),p,t,delta,poses,low,ref]:a.tofile(f)
        got=np.loadtxt(StringIO(subprocess.check_output([engine,lib,fixture],text=True))).reshape(n,2,5)
    stats={}
    for i,(name,(poly,z)) in enumerate(zip(cases,expect)):
        actual=got[i]
        if poly is None:
            if not (np.isposinf(actual[:,0]).all() and (actual[:,1:]==0).all()):raise ValueError((name,actual))
            stats[name]={'has_area':False};continue
        center,v=poly;err=max(abs(actual[0,0]-center@z),abs(actual[1,0]-min(v@z)))
        if err>2e-9 or max(abs(actual[0,1:4]-center))>2e-5 or max(abs(actual[:,1:].sum(1)-1))>1e-5:raise ValueError((name,actual,center,z,err))
        stats[name]={'has_area':True,'maximum_gap_error_m':float(err),'centroid_weight_error':float(max(abs(actual[0,1:4]-center)))}
    out.write_text(json.dumps(dict(kernel='nm_pelvic_finite_surface',metallib=sha(lib),cases=stats),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
