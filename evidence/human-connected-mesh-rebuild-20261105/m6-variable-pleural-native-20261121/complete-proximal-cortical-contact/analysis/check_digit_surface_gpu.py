"""Compare the actual Metal triangle query with independent Minkowski hull distances."""
import json,subprocess,sys
import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

def ref(a,b):
    # For a full-dimensional Minkowski difference the origin is inside iff the
    # real finite triangles intersect. Inside, its nearest hull plane gives the depth.
    pts=(a[:,None]-b).reshape(-1,3);h=ConvexHull(pts)
    z=h.equations[:,3]
    if max(z)<=0:return float(max(z))
    vals=[]
    for face in pts[h.simplices]:
        q,a1,a2=face;A=np.stack([a1-q,a2-q],axis=1)
        x=np.linalg.lstsq(A,-q,rcond=None)[0]
        if min(x)>=0 and sum(x)<=1:vals.append(np.linalg.norm(q+A@x))
        for i in range(3):
            p=face[i];d=face[(i+1)%3]-p
            vals.append(np.linalg.norm(p+d*np.clip(-p@d/(d@d),0.,1.)))
    return min(vals)
def main(root,dst):
    dst.mkdir(exist_ok=True);shader=(root/'digit_surface_query.metalinc').read_text().split('kernel void nm_digit_surface_faces')[0]
    s=dst/'coupon.metal';s.write_text('#include <metal_stdlib>\nusing namespace metal;\n'+shader+'''
kernel void nm_digit_surface_coupon(device const float4 *x [[buffer(0)]],device float4 *out [[buffer(1)]],uint k [[thread_position_in_grid]]){
    float3 a[3],b[3];for(uint j=0;j<3;++j){a[j]=x[6*k+j].xyz;b[j]=x[6*k+3+j].xyz;}
    float3 p,q,n;float g;bool ok=nmDigitTriangles(a,b,INFINITY,g,p,q,n);
    out[3*k]=float4(p,ok?g:INFINITY);out[3*k+1]=float4(q,0);out[3*k+2]=float4(n,0);
}
''')
    air=dst/'coupon.air';lib=dst/'coupon.metallib';exe=dst/'test'
    subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-c',s,'-o',air],check=True)
    subprocess.run(['xcrun','-sdk','macosx','metallib',air,'-o',lib],check=True)
    subprocess.run(['xcrun','clang++','-O2','-std=c++20','-fobjc-arc',root/'check_digit_surface_gpu.mm','-framework','Metal','-framework','Foundation','-o',exe],check=True)
    rng=np.random.default_rng(94087);tris=[];expected=[]
    for i in range(192):
        x=rng.normal(size=(2,3,3))*.002
        if i%2:x[1]+=rng.normal(size=3)*.001
        x+=rng.uniform(-.025,.025,size=3)
        x=x.astype('<f4');tris.append(x);expected.append(ref(*x.astype(float)))
    a=np.array([[0,0,0],[.003,0,0],[0,.003,0]],dtype='<f4')
    for b,g in [(a+[0,0,.002],.002),(a+[.004,0,0],.001),(a+np.array([.0001,.0001,0]),0)]:
        tris.append(np.stack([a,b]).astype('<f4'));expected.append(g)
    points=np.array(tris);src=dst/'input.bin';out=dst/'out.bin';packed=np.zeros((len(tris),2,3,4),dtype='<f4');packed[:,:,:,:3]=points;packed.tofile(src)
    subprocess.run([exe,lib,src,out],check=True)
    v=np.fromfile(out,'<f4').reshape(-1,3,4).astype(float);g=v[:,0,3];n=v[:,2,:3]
    gaperr=np.abs(g-expected);line=np.linalg.norm(v[:,0,:3]-v[:,1,:3]-g[:,None]*n,axis=1)
    data=dict(cases=len(g),penetrating=sum(float(z)<-1.e-8 for z in expected),max_signed_distance_error_m=float(gaperr.max()),
        max_witness_offset_error_m=float(line.max()),max_unit_normal_error=float(np.abs(np.linalg.norm(n,axis=1)-1).max()))
    (dst/'comparison.json').write_text(json.dumps(data,indent=2)+'\n');print(data)
    if not np.isfinite(v).all() or gaperr.max()>2.e-7 or line.max()>2.e-7 or data['max_unit_normal_error']>1.e-4:raise ValueError('GPU signed distance or wrench line differs from the reference')
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
