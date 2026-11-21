#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
static void check(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
int main(int argc,char** argv){@autoreleasepool{
    if(argc!=2)return 2;
    auto device=MTLCreateSystemDefaultDevice();NSError* error=nil;
    auto lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
    auto pipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nm_pelvic_incremental_compression"] error:&error];
    check(pipeline!=nil,"kernel");auto queue=[device newCommandQueue];
    auto make=[&](size_t n){auto b=[device newBufferWithLength:n options:MTLResourceStorageModeShared];check(b!=nil,"allocation");std::memset(b.contents,0,n);return b;};
    std::array<id<MTLBuffer>,9> b{make(512*4),make(512*64),make(2*65536*4),make(8),make(512*4),make(512*4),make(2*16),make(512*16),make(512*112)};
    auto slots=(unsigned*)b[0].contents;std::memset(slots,0xff,b[0].length);
    auto rows=(float*)b[1].contents,W=(float*)b[2].contents,p=(float*)b[4].contents,dp=(float*)b[5].contents;
    auto N=(float*)b[7].contents;auto sites=(unsigned*)b[8].contents;
    ((unsigned*)b[3].contents)[0]=((unsigned*)b[3].contents)[1]=1;
    for(unsigned j:{0u,1u,256u,257u}){slots[j]=j;N[4*j]=1;sites[28*j+20]=j/256;((float*)sites)[28*j+24]=1;}
    // Side zero has redundant identical rows. Side one first releases a
    // previous compression and then adds load against a coupled free predictor.
    W[0]=W[1]=W[256]=W[257]=1;W[65536]=W[65536+257]=1;W[65536+1]=W[65536+256]=.25f;
    rows[3]=rows[19]=-.001f;rows[256*16+3]=.007f;rows[257*16+3]=-.002f;
    p[256]=.003f;((float*)b[6].contents)[4]=.0004f;float damping=.5f;
    auto cb=[queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:pipeline];
    for(unsigned j=0;j<b.size();++j)[e setBuffer:b[j] offset:0 atIndex:j];[e setBytes:&damping length:4 atIndex:9];
    [e dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];
    check(cb.status==MTLCommandBufferStatusCompleted,"dispatch");
    check(std::fabs(2*(p[0]+p[1])-.001)<2e-9,"dependent compression");
    check(std::fabs(p[256]-.0015)<1e-9,"load must release");
    check(std::fabs(dp[256]+.0015)<1e-9,"release increment");
    check(std::fabs(2*p[257]-.00315)<2e-9,"free predictor and coupled load");
    std::cout<<"redundant_rows,release,force_predictor,shared_relaxation\npass,pass,pass,pass\n";
}}
