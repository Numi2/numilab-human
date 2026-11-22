#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <initializer_list>
#include <cstring>

int main(int argc, const char** argv) {
    @autoreleasepool {
        if (argc != 2) return 2;
        id<MTLDevice> gpu = MTLCreateSystemDefaultDevice();
        NSError* error = nil;
        id<MTLLibrary> library = [gpu newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
        if (!library) return 3;
        auto pipeline = [&](NSString* name) {
            auto function = [library newFunctionWithName:[@"numi_matter_metal::nm_fem_registered_release_" stringByAppendingString:name]];
            return [gpu newComputePipelineStateWithFunction:function error:&error];
        };
        auto gate = pipeline(@"bed_begin_gate");
        auto clear = pipeline(@"reset");
        if (!gate || !clear) return 4;
        auto word = [gpu newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        auto stats = [gpu newBufferWithLength:16 options:MTLResourceStorageModeShared];
        auto queue = [gpu newCommandQueue];
        unsigned cases = 0;
        for (bool hasViolation : {false,true}) for (bool admissible : {false,true}) for (bool reset : {false,true}) {
            *(uint32_t*)word.contents = hasViolation ? 28878u : UINT32_MAX;
            const float orig[4] = {1.e-5f,3.f,float(admissible),-7.f};
            std::memcpy(stats.contents,orig,sizeof(orig));
            auto cb=[queue commandBuffer]; auto enc=[cb computeCommandEncoder];
            if (reset) {
                [enc setComputePipelineState:clear];[enc setBuffer:word offset:0 atIndex:0];
                [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [enc setComputePipelineState:gate];[enc setBuffer:word offset:0 atIndex:0];[enc setBuffer:stats offset:0 atIndex:1];
            [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [enc endEncoding];[cb commit];[cb waitUntilCompleted];
            const float* actual=(const float*)stats.contents;
            if (cb.status!=MTLCommandBufferStatusCompleted || actual[2]!=float(admissible && (reset || !hasViolation)) ||
                actual[0]!=orig[0] || actual[1]!=orig[1] || actual[3]!=orig[3]) return 5;
            ++cases;
        }
        std::printf("{\"device\":\"%s\",\"passing_cases\":%u}\n",gpu.name.UTF8String,cases);
    }
}
