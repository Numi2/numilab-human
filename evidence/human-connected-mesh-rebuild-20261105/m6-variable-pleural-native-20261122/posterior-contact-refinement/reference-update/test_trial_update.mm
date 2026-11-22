#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <cstring>
struct alignas(16) Row {float hi,lo,velocity,delta;};
int main(int argc,const char** argv) {
    @autoreleasepool {
        if (argc!=3) return 2;
        const bool incremental=std::strcmp(argv[2],"incremental")==0;
        id<MTLDevice> gpu=MTLCreateSystemDefaultDevice(); NSError* error=nil;
        auto library=[gpu newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
        auto fn=[library newFunctionWithName:@"check_material_update"];
        auto pipeline=fn?[gpu newComputePipelineStateWithFunction:fn error:&error]:nil;
        if (!pipeline) { std::fprintf(stderr,"%s\n",[[error description] UTF8String]); return 2; }
        auto queue=[gpu newCommandQueue];auto cmd=[queue commandBuffer];auto encoder=[cmd computeCommandEncoder];
        auto data=[gpu newBufferWithLength:7*sizeof(Row) options:MTLResourceStorageModeShared];
        [encoder setComputePipelineState:pipeline];[encoder setBuffer:data offset:0 atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(7,1,1) threadsPerThreadgroup:MTLSizeMake(7,1,1)];
        [encoder endEncoding];[cmd commit];[cmd waitUntilCompleted];
        if (cmd.status!=MTLCommandBufferStatusCompleted) return 3;
        const auto* rows=static_cast<const Row*>([data contents]);double step=std::ldexp(double(0.0012f),-33);
        for(int i=0;i<7;++i){
            const double measured=double(rows[i].hi)+rows[i].lo;
            const double multiple=i==0?1.:i==1?2.:0.;
            const double target=incremental?multiple*step:0.;
            std::printf("%d %.12g %.12g %.12g %.12g\n",i,measured,target,double(rows[i].velocity),double(rows[i].delta));
            if(!std::isfinite(measured) || std::abs(measured-target)>6.e-18 || rows[i].velocity!=.75f) return 4;
            if(incremental && (i==0 || i==1) && std::abs(double(rows[i].delta)-std::ldexp(target,16))>4.e-13) return 5;
        }
        return 0;
    }
}
