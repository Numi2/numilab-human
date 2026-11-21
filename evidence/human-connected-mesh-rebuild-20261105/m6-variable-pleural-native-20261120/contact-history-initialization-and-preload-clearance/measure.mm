#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <libproc.h>
#include <limits>
#include <span>
#include <string>
#include <vector>
#include <unistd.h>
#include "numi/matter/shared.h"

static std::string errorString(NSError* error) {
    return error != nil ? error.localizedDescription.UTF8String : "Metal command failed";
}

#include "upload_plan_fixture.hpp"

static void require(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

static uint64_t footprint() {
    struct rusage_info_v2 info {};
    require(proc_pid_rusage(getpid(), RUSAGE_INFO_V2, (rusage_info_t*)&info)==0,
            "Cannot read native process footprint");
    return info.ri_phys_footprint;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        require(argc == 4, "Expected mode, count and repeats");
        const bool control = std::strcmp(argv[1], "control") == 0;
        const auto count = std::strtoull(argv[2], nullptr, 10);
        const auto repeat = std::strtoull(argv[3], nullptr, 10);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        require(device != nil && queue != nil, "Apple GPU unavailable");
        bool valid = true;
        std::size_t allocated = 0;
        std::string error;
        const auto before = footprint();
        const auto started = std::chrono::steady_clock::now();
        UploadPlan upload(device, queue);
        id<MTLBuffer> buffer;
        std::vector<NMDeformableContactHistoryGPU> initial;
        if (control) {
            initial.resize(count);
            buffer = upload.repeated(std::span<const NMDeformableContactHistoryGPU>(initial),
                                     repeat, valid, allocated);
        } else {
            buffer = upload.zeroed<NMDeformableContactHistoryGPU>(
                count, repeat, valid, allocated);
        }
        require(valid && buffer != nil && upload.finish(error), error.c_str());
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        const auto after = footprint();
        const auto bytes = count*repeat*sizeof(NMDeformableContactHistoryGPU);
        require(buffer.length == std::max<std::size_t>(bytes,sizeof(NMDeformableContactHistoryGPU)),
                "Unexpected typed size");
        require(buffer.length == allocated, "Incorrect resident byte counter");

        constexpr std::size_t chunk = 4*1024*1024;
        id<MTLBuffer> verify = [device newBufferWithLength:std::min<std::size_t>(chunk,buffer.length)
            options:MTLResourceStorageModeShared];
        require(verify != nil, "Readback allocation failed");
        const std::size_t verifyBytes = control ? bytes : buffer.length;
        for (size_t offset=0; offset<verifyBytes; offset+=chunk) {
            const auto size=std::min(chunk,verifyBytes-offset);
            std::memset(verify.contents, 0xa5, size);
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
            [blit copyFromBuffer:buffer sourceOffset:offset toBuffer:verify destinationOffset:0 size:size];
            [blit endEncoding]; [command commit]; [command waitUntilCompleted];
            require(command.status==MTLCommandBufferStatusCompleted, "GPU readback failed");
            const auto* contents=static_cast<const unsigned char*>(verify.contents);
            require(std::all_of(contents, contents+size, [](unsigned char byte) {return byte == 0;}),
                    "Nonzero history byte");
        }
        if (!control) {
            bool okay=true;
            std::size_t reported=0;
            UploadPlan invalid(device,queue);
            require(invalid.zeroed<NMDeformableContactHistoryGPU>(
                        std::numeric_limits<std::size_t>::max(), 2, okay, reported) == nil &&
                    !okay && reported == 0, "Count overflow accepted");
            okay=true;
            require(invalid.zeroed<NMDeformableContactHistoryGPU>(
                        std::numeric_limits<std::size_t>::max(), 1, okay, reported) == nil &&
                    !okay && reported == 0, "Byte overflow accepted");
            okay=true;
            require(upload.zeroed<NMDeformableContactHistoryGPU>(
                1,1,okay,reported)==nil && !okay && reported==0,
                "Append after finish accepted");
        }
        const auto delta=static_cast<long long>(after)-static_cast<long long>(before);
        std::printf("{\"mode\":\"%s\",\"GPU\":\"%s\",\"count\":%llu,"
                    "\"repeat\":%llu,\"checked_bytes\":%zu,\"private_bytes\":%zu,"
                    "\"footprint_delta_bytes\":%lld,\"upload_wall_ms\":%.4f}\n",
                    argv[1], device.name.UTF8String, count, repeat,
                    static_cast<std::size_t>(verifyBytes), allocated, delta, elapsed);
    }
}
