// Evaluate prescribed material points at the release pose using the same paired
// floating-root kinematics and Metal projection used by the live tissue adapter.
#pragma once

static std::vector<nm_float4> registeredReleaseTargets(
    const metalrobo::EngineModel& model, std::span<const float> q,
    std::span<const NMNumiHumanTendonFEMNodeAnchorGPU> anchors,
    const std::filesystem::path& metallib) {
    require(!anchors.empty() && q.size()>=7,"Missing registered release state");
    MRCompensatedRootTranslationGPU root{};
    std::string error;
    require(metalrobo::makeNumiHumanInitialRootTranslation({q[0],q[1],q[2]},root,error),error);
    metalrobo::MetalArticulatedOperatorInput input;
    input.articulationIndex=0u;
    input.environmentCount=1u;
    input.q=q;
    input.rootTranslations=std::span(&root,1);
    metalrobo::MetalArticulatedOperatorConfig config;
    config.pointJacobiansOnly=true;
    metalrobo::MetalArticulatedOperatorResult fk;
    const auto status=metalrobo::runMetalArticulatedOperator(model,input,fk,config);
    require(status.succeeded() && fk.bodyPoses.size()==model.bodies.size(),
            "Registered release FK failed: "+status.message);
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    NSError* err=nil;
    id<MTLLibrary> library=[device newLibraryWithURL:
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib.c_str()]] error:&err];
    id<MTLFunction> kernel=[library newFunctionWithName:
        @"numi_matter_metal::nm_numi_human_assemble_fem_kinematic_targets"];
    id<MTLComputePipelineState> pipeline=kernel?[device newComputePipelineStateWithFunction:kernel error:&err]:nil;
    require(pipeline!=nil,"Cannot load registered release target projection");
    NMNumiHumanTendonFEMLoadDispatchGPU dispatch{};
    dispatch.environmentCount=1;
    dispatch.femNodeCount=static_cast<uint32_t>(anchors.size());
    dispatch.bodyPoseStride=static_cast<uint32_t>(fk.bodyPoses.size());
    const auto mode=MTLResourceStorageModeShared;
    id<MTLBuffer> src=[device newBufferWithBytes:anchors.data() length:anchors.size_bytes() options:mode];
    id<MTLBuffer> poses=[device newBufferWithBytes:fk.bodyPoses.data()
        length:fk.bodyPoses.size()*sizeof(fk.bodyPoses.front()) options:mode];
    id<MTLBuffer> out=[device newBufferWithLength:anchors.size()*sizeof(nm_float4) options:mode];
    require(src && poses && out,"Registered release projection allocation failed");
    id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:0];
    [encoder setBuffer:src offset:0 atIndex:1];
    [encoder setBuffer:poses offset:0 atIndex:2];
    [encoder setBuffer:out offset:0 atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(anchors.size(),1,1)
        threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth,1,1)];
    [encoder endEncoding];[command commit];[command waitUntilCompleted];
    require(command.status==MTLCommandBufferStatusCompleted,"Registered release projection failed");
    const auto* targets=static_cast<const nm_float4*>(out.contents);
    return {targets,targets+anchors.size()};
}


struct RegisteredAttachmentRelease {
    std::vector<uint32_t> ids;
    std::vector<nm_float4> targets;
};

static RegisteredAttachmentRelease prepareRegisteredAttachmentRelease(
    const metalrobo::EngineModel& model, std::span<const float> q,
    const auto& owner, const std::filesystem::path& metallib, auto&& appendSource) {
    RegisteredAttachmentRelease release;
    std::vector<NMNumiHumanTendonFEMNodeAnchorGPU> active;
    for(size_t i=0;i<owner.nodeAnchors.size();++i) {
        const auto& anchor=owner.nodeAnchors[i];
        if(anchor.flags!=NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE) continue;
        require(i<=UINT32_MAX && anchor.bodyIndex<model.bodies.size(),"Unknown release attachment");
        release.ids.push_back(static_cast<uint32_t>(i));
        active.push_back(anchor);
    }
    release.targets=registeredReleaseTargets(model,q,active,metallib);
    appendSource("registered_release_node_ids_v1",std::as_bytes(std::span(release.ids)));
    appendSource("registered_release_world_targets_v1",std::as_bytes(std::span(release.targets)));
    return release;
}

static void installRegisteredAttachmentRelease(const RegisteredAttachmentRelease& release,
    numi::matter::RuntimeStateSnapshot& initial, numi::matter::Runtime& runtime) {
    require(initial.available && release.targets.size()==release.ids.size(),"Invalid release checkpoint");
    double maximum=0.0;
    for(size_t i=0;i<release.targets.size();++i) {
        const auto target=release.targets[i];
        auto& node=initial.femNodes.at(release.ids[i]);
        auto& pos=node.positionAndMass;
        auto& h=node.referenceDisplacementAndMode;
        auto& l=node.referenceDisplacementLow;
        const auto rest=node.restAndFixed, v=node.velocityAndInverseMass;
        require(target.w==1 && std::isfinite(target.x) && std::isfinite(target.y) && std::isfinite(target.z) &&
                rest.w>0 && pos.w>0 && v.x==0 && v.y==0 && v.z==0 && v.w==0,
                "Release initialization requires a stationary prescribed tissue point");
        maximum=std::max(maximum,std::hypot(double(target.x)-pos.x,double(target.y)-pos.y,double(target.z)-pos.z));
        if(h.w==1) {
            auto split=[](float target,float origin,float& high,float& low){
                const double diff=double(target)-origin;
                high=static_cast<float>(diff);
                low=static_cast<float>(diff-double(high));
            };
            split(target.x,rest.x,h.x,l.x);
            split(target.y,rest.y,h.y,l.y);
            split(target.z,rest.z,h.z,l.z);
        }
        pos.x=target.x;pos.y=target.y;pos.z=target.z;
    }
    const auto replay=runtime.restore(initial);
    require(replay.encoded,"Registered release failed native checkpoint admission: "+replay.message);
    std::cout<<"registered_attachment_release=compiled fixed_nodes="<<release.ids.size()
             <<" maximum_source_offset_m="<<maximum<<" source_reference_and_mass=immutable"
             <<" physical_time_advanced=0"<<std::endl;
}
