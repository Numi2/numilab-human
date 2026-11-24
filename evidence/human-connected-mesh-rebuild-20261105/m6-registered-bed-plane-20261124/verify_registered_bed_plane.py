"""Exercise the exact active shader fragment on the GPU and compare with an independent plane projection."""
from pathlib import Path
import hashlib,json,subprocess,struct
import numpy as np
from scipy.spatial.transform import Rotation
def check():
    d=Path('joined-plane-unit').resolve();d.mkdir(exist_ok=True);src=Path('joined-registered-plane-owner/human_respiration.metal')
    text=src.read_text();i=text.index('    if (m.deformationKind == 3u && d.w != 0u && stock > 0.0f)');a=text[i:text.index('    vertices[i].position=float4(p,1);',i)]
    gpu='''#include <metal_stdlib>
using namespace metal;
struct Map {uint deformationKind;};
kernel void testplane(device const float4* source [[buffer(0)]],device float4* result [[buffer(1)]],uint row [[thread_position_in_grid]]) {
 uint j=4*row;float3 p=source[j].xyz,n=source[j+1].xyz;float stock=source[j].w;
 float4 supportPlane=source[j+2];Map m={uint(source[j+3].x)};uint4 d=uint4(0,0,0,uint(source[j+3].y));
'''+a+'''result[2*row]=float4(p,0);result[2*row+1]=float4(n,0);
}'''
    (d/'shader.metal').write_text(gpu)
    host='''#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
int main(int argc,const char** argv) { @autoreleasepool {
if(argc!=4)return 1;
id<MTLDevice> device=MTLCreateSystemDefaultDevice();NSError* err=nil;
id<MTLLibrary> lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
if(!lib){NSLog(@"%@",err);return 2;}
id<MTLComputePipelineState> ps=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"testplane"] error:&err];
NSData* data=[NSData dataWithContentsOfFile:@(argv[2])];
if(!ps || !data || data.length%64!=0 || data.length==0)return 3;
id<MTLBuffer> input=[device newBufferWithBytes:data.bytes length:data.length options:MTLResourceStorageModeShared];
id<MTLBuffer> output=[device newBufferWithLength:data.length/2 options:MTLResourceStorageModeShared];
id<MTLCommandQueue> q=[device newCommandQueue];id<MTLCommandBuffer> cmd=[q commandBuffer];
id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];[enc setComputePipelineState:ps];[enc setBuffer:input offset:0 atIndex:0];[enc setBuffer:output offset:0 atIndex:1];
[enc dispatchThreads:MTLSizeMake(data.length/64,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
[enc endEncoding];[cmd commit];[cmd waitUntilCompleted];if(cmd.error){NSLog(@"%@",cmd.error);return 4;}
if(![[NSData dataWithBytes:output.contents length:output.length] writeToFile:@(argv[3]) atomically:YES])return 5;
std::puts([[device name] UTF8String]);return 0;
}}'''
    (d/'test.mm').write_text(host)
    for c in [['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-c','shader.metal','-o','shader.air'],['xcrun','-sdk','macosx','metallib','shader.air','-o','test.metallib'],['xcrun','clang++','-std=c++17','-fobjc-arc','test.mm','-framework','Metal','-framework','Foundation','-o','tester']]:subprocess.run(c,cwd=d,check=True)
    cfg=json.load(open('joined-load-motion-source/launch.json'));p=Path(cfg['argv'][cfg['argv'].index('--support-contact-payload')+1]);h=struct.unpack_from('<8s4I32s7f',p.read_bytes());origin=np.asarray(h[6:9]);N=np.asarray(h[9:12]);N=N/np.linalg.norm(N)
    data=[];expected=[];flat_count=0
    for n,o in [(np.array([0,0,1.0]),np.zeros(3)),(N,origin),(np.array([.4,0,np.sqrt(.84)]),np.array([.2,-.1,.36]))]:
        t=np.cross(n,[0,1,0]);t=t/np.linalg.norm(t)
        for gap in [-.06,-.03,-.001,.007]:
            for face,mode,enabled in [(1,3,1),(-1,3,1),(-1,3,0),(-1,1,1)]:
                pp=o+.04*t+gap*n;normal=face*n;stock=.05
                data.append([[*pp,stock],[*normal,0],[*n,-o@n],[mode,enabled,0,0]])
                proj=gap<0 and gap>=-stock and mode==3 and enabled
                expected.append([o+.04*t if proj else pp,-n if proj and face<0 else normal])
        if n[2]==1:flat_count=len(expected)
    x=np.array(data,dtype='<f4');x.tofile(d/'in.f32');device=subprocess.check_output([str(d/'tester'),str(d/'test.metallib'),str(d/'in.f32'),str(d/'out.f32')],text=True).strip()
    got=np.fromfile(d/'out.f32',dtype='<f4').reshape(-1,2,4)[:,:,:3].copy();ref=np.array(expected);error=np.max(abs(got-ref),axis=(0,1))
    flat=np.asarray(ref[:flat_count],dtype='<f4');assert np.array_equal(got[:flat_count],flat)
    if np.max(error)>1e-7:raise ValueError('Metal projection exceeds 0.1 micrometre')
    r=dict(device=device,rows=len(expected),flat_bitwise_rows=flat_count,loaded_free_disabled_and_beyond_stock=True,maximum_vector_component_error_m=float(np.max(error)),shader_sha256=hashlib.sha256(text.encode()).hexdigest(),scope='Exact selected reconstruction fragment exercised on GPU. Integrated trial separately tests the native entry point and physical forces.')
    (d/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
if __name__=='__main__':check()
