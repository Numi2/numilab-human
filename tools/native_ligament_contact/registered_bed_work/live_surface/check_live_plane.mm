#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <simd/simd.h>
int main(int argc,const char** argv){
    @autoreleasepool{
        if(argc!=5)return 2;
        auto path=[&](int n){return [NSString stringWithUTF8String:argv[n]];};
        NSError* error=nil;auto gpu=MTLCreateSystemDefaultDevice();
        auto library=[gpu newLibraryWithURL:[NSURL fileURLWithPath:path(1)] error:&error];
        if(!library)return 3;
        auto first=[library newFunctionWithName:@"make_bed_test_state"];
        auto second=[library newFunctionWithName:@"test_live_bed_rows"];
        auto prepare=[gpu newComputePipelineStateWithFunction:first error:&error];
        auto test=[gpu newComputePipelineStateWithFunction:second error:&error];
        if(!prepare || !test)return 4;
        auto load=[&](NSString* file){auto url=[path(2) stringByAppendingPathComponent:file];
            auto data=[NSData dataWithContentsOfFile:url];return data.length?[gpu newBufferWithBytes:data.bytes length:data.length options:MTLResourceStorageModeShared]:nil;};
        auto shared=[&](NSUInteger bytes){return [gpu newBufferWithLength:bytes options:MTLResourceStorageModeShared];};
        auto nodes=load(@"nodes"),accepted=load(@"accepted"),direction=load(@"direction"),params=load(@"params");
        if(!nodes || !accepted || nodes.length!=accepted.length || direction.length!=nodes.length || params.length!=8 || nodes.length%16)return 5;
        const NSUInteger n=nodes.length/16;
        // Shader defines the stride. Reserve more than a full shared FEM node per probe.
        auto current=shared(n*512),old=shared(n*512),records=shared(n*16);
        auto corners=load(@"corners");if(!corners)return 6;
        auto flags=shared(4);*static_cast<uint32_t*>(flags.contents)=1;
        auto faces=load(@"faces"),measure=load(@"measure"),ranges=load(@"range");
        auto off=shared(2*corners.length);
        auto arg=[second newArgumentEncoderWithBufferIndex:0];
        auto table=shared(arg.encodedLength);[arg setArgumentBuffer:table offset:0];
        id<MTLBuffer> a[]={flags,faces,measure,ranges,corners,off};
        for(unsigned k=0;k<6;++k){if(!a[k])return 7;[arg setBuffer:a[k] offset:0 atIndex:20+k];}
        auto q=[gpu newCommandQueue];auto cb=[q commandBuffer];auto e=[cb computeCommandEncoder];
        [e setComputePipelineState:prepare];
        id<MTLBuffer> src[]={nodes,accepted},dst[]={current,old};
        for(unsigned i=0;i<2;++i){
            [e setBuffer:src[i] offset:0 atIndex:0];[e setBuffer:dst[i] offset:0 atIndex:1];
            [e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];}
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];[e setComputePipelineState:test];
        for(unsigned k=0;k<6;++k)[e useResource:a[k] usage:(k==5?(MTLResourceUsageRead|MTLResourceUsageWrite):MTLResourceUsageRead)];
        id<MTLBuffer> bindings[]={table,current,old,direction,records,params};
        for(unsigned k=0;k<6;++k)[e setBuffer:bindings[k] offset:0 atIndex:k];
        [e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e endEncoding];[cb commit];[cb waitUntilCompleted];
        if(cb.status!=MTLCommandBufferStatusCompleted){std::fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);return 8;}
        auto data=[NSData dataWithBytesNoCopy:records.contents length:records.length freeWhenDone:NO];
        if(![data writeToFile:path(3) options:NSDataWritingAtomic error:&error])return 9;
        std::printf("%s %lu node responses\n",gpu.name.UTF8String,(unsigned long)n);
    }
}
