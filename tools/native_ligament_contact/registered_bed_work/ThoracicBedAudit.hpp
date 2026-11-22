#pragma once
#include <charconv>
#include <fstream>
#include <iomanip>

// Optional sampled observation of the physical stand solver. Only the
// accepted export path can release records to the caller.
struct ThoracicBedAudit {
    static constexpr unsigned stride = 32;
    unsigned endStep=0, rows=0, vertices=0;
    NSUInteger mapOffset=0;
    id<MTLComputePipelineState> pipeline=nil;
    id<MTLBuffer> selection=nil, mapping=nil, anatomy=nil, samples=nil;

    void prepare(id<MTLDevice> device, id<MTLLibrary> library, id<MTLBuffer> chosen,
                 id<MTLBuffer> map, NSUInteger offset, id<MTLBuffer> anatomyBuffer,
                 unsigned count, unsigned vertexCount) {
        const char* text=std::getenv("NUMI_HUMAN_THORACIC_BED_AUDIT_END_STEP");
        if(!text) return;
        const char* end=text+std::strlen(text);
        const auto parsed=std::from_chars(text,end,endStep);
        if(parsed.ec!=std::errc{} || parsed.ptr!=end || !endStep || endStep>32000u ||
           endStep%stride || !count || count>128u || !vertexCount || !chosen ||
           chosen.length<count*sizeof(unsigned) || !map || offset>map.length ||
           vertexCount>(map.length-offset)/sizeof(MRHumanRestingVertexMap) ||
           !anatomyBuffer || anatomyBuffer.length<sizeof(MRHumanRestingAnatomyGPU))
            throw std::runtime_error("invalid registered bed work audit domain");
        NSError* error=nil;
        pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"nm_human_thoracic_bed_work"] error:&error];
        rows=count;vertices=vertexCount;selection=chosen;mapping=map;mapOffset=offset;anatomy=anatomyBuffer;
        samples=[device newBufferWithLength:(endStep/stride)*rows*8u*sizeof(mr_float4)
                                  options:MTLResourceStorageModeShared];
        if(!samples || !pipeline) throw std::runtime_error("could not prepare thoracic bed work audit");
        std::memset(samples.contents,0,samples.length);
    }

    void encode(const metalrobo::MetalNumanXTransactionPass& p,id<MTLBuffer> capturedBodies,
                id<MTLBuffer> capturedState) {
        const unsigned step=p.stepIndex+1;
        if(!samples || step>endStep || step%stride) return;
        if(p.phase!=metalrobo::MetalNumanXTransactionPhase::postDynamics || p.environmentCount!=1 ||
           !p.standContactEnabled || p.standContactImpulseSampleStepIndex!=p.stepIndex ||
           p.standContactCount!=rows || p.standContactImpulseOffset>UINT32_MAX ||
           p.standContactImpulseOffset+3u*rows>p.standVectorElementCount ||
           p.pointCount>UINT32_MAX || p.bodyCount>UINT32_MAX || !p.pointPositionLow ||
           p.pointPositionLowElementCount<p.pointCount || p.pointWorldElementCount<p.pointCount ||
           !p.standVectorWorkspace || !p.pointWorld || !p.standContacts || !p.standStatuses)
            throw std::runtime_error("missing current candidate bed impulse or pre-step geometry");
        const mr_uint4 domain{p.stepIndex,rows,static_cast<unsigned>(p.standContactImpulseOffset),step/stride-1};
        const mr_uint4 bounds{vertices,static_cast<unsigned>(p.pointCount),static_cast<unsigned>(p.bodyCount),0};
        auto command=(__bridge id<MTLCommandBuffer>)p.commandBuffer;
        auto enc=[command computeCommandEncoder];
        if(!enc) throw std::runtime_error("missing bed work encoder");
        [enc setComputePipelineState:pipeline]; [enc setBytes:&domain length:sizeof(domain) atIndex:0];
        [enc setBytes:&bounds length:sizeof(bounds) atIndex:1];[enc setBuffer:anatomy offset:0 atIndex:2];
        [enc setBuffer:mapping offset:mapOffset atIndex:3];[enc setBuffer:selection offset:0 atIndex:4];
        [enc setBuffer:(__bridge id<MTLBuffer>)p.standContacts offset:0 atIndex:5];
        [enc setBuffer:(__bridge id<MTLBuffer>)p.standVectorWorkspace offset:0 atIndex:6];
        [enc setBuffer:(__bridge id<MTLBuffer>)p.pointWorld offset:0 atIndex:7];
        [enc setBuffer:(__bridge id<MTLBuffer>)p.pointPositionLow offset:0 atIndex:8];
        [enc setBuffer:capturedBodies offset:0 atIndex:9];[enc setBuffer:capturedState offset:0 atIndex:10];
        [enc setBuffer:(__bridge id<MTLBuffer>)p.standStatuses offset:0 atIndex:11];
        [enc setBuffer:samples offset:0 atIndex:12];
        [enc setBytes:&p.standGroundNormal length:sizeof(p.standGroundNormal) atIndex:13];
        [enc dispatchThreads:MTLSizeMake(rows,1,1) threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth,1,1)];
        [enc endEncoding];
    }

    void write(const std::filesystem::path& dir,unsigned acceptedStep) const {
        if(!samples || acceptedStep!=endStep) return;
        std::ofstream file(dir/"resting-thoracic-bed-work.csv");
        file<<"accepted_step,region,vertex,source,local_x,local_y,local_z,weight,qx,qy,qz,qw,qd,qr,vd,vr,ix,iy,iz,basal,jdx,jdy,jdz,id_pa_s,jrx,jry,jrz,ir_pa_s,dx,dy,dz,work_j\n"<<std::setprecision(9);
        const auto* data=static_cast<const unsigned char*>(samples.contents);
        for(unsigned t=0;t<endStep/stride;++t) for(unsigned r=0;r<rows;++r) {
            unsigned ids[4];float values[28];
            const auto* src=data+(t*rows+r)*128u;
            std::memcpy(ids,src,16);std::memcpy(values,src+16,112);
            if(ids[0]!=(t+1)*stride || ids[1]!=r) throw std::runtime_error("missing accepted thoracic bed audit row");
            file<<ids[0]<<','<<ids[1]<<','<<ids[2]<<','<<ids[3];
            for(float x:values) {
                if(!std::isfinite(x)) throw std::runtime_error("invalid thoracic bed witness");
                file<<','<<x;
            }
            file<<'\n';
        }
        file.close();if(!file) throw std::runtime_error("could not write accepted thoracic bed work");
    }
};
