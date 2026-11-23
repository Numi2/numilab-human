    void certifyBedEndpoint(const metalrobo::MetalNumanXTransactionPass& pass) {
        if(!resolvedBed_ || pass.bodyCount!=validatedBodyCount_ || pass.environmentCount!=1u ||
           pass.bodyPoseStride!=validatedBodyCount_ || pass.articulationFirstBody!=validatedFirstBody_ ||
           !pass.bodyPoses || !pass.bodyPositionLow || !pass.standStatuses)
            throw std::invalid_argument("Invalid endpoint bed pose lease");
        auto cmd=(__bridge id<MTLCommandBuffer>)pass.commandBuffer;
        auto bodies=(__bridge id<MTLBuffer>)pass.bodyPoses;
        auto low=(__bridge id<MTLBuffer>)pass.bodyPositionLow;
        MRHumanRestingSupportDispatchGPU d{};
        d.counts={vertexCount_,regionCount_,static_cast<unsigned>(pass.dofCount),1};
        d.strides.x=validatedBodyCount_;
        d.identity={validatedFirstBody_,influenceCount_,regionCount_,MR_NUMI_HUMAN_SUPPORT_GEOMETRY_ABI_VERSION};
        auto enc=[cmd computeCommandEncoder];
        if(!enc)throw std::runtime_error("Missing endpoint orientation encoder");
        [enc setComputePipelineState:orientationPipeline_];
        [enc setBytes:&d length:sizeof(d) atIndex:0];
        [enc setBuffer:bodies offset:0 atIndex:1];
        [enc setBuffer:normalizedOrientations_ offset:0 atIndex:2];
        [enc setBuffer:rotationZBasis_ offset:0 atIndex:3];
        [enc setBuffer:low offset:0 atIndex:4];
        [enc dispatchThreads:MTLSizeMake(validatedBodyCount_,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [enc endEncoding];
        resolvedBed_->certifyEndpoint(cmd,d,vertexMap_,vertexMapOffsetBytes_,influences_,bodies,low,
            rotationZBasis_,(__bridge id<MTLBuffer>)pass.standStatuses,pass.stepIndex);
    }
