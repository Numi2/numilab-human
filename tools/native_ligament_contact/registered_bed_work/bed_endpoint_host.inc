    void certifyEndpoint(id<MTLCommandBuffer> cmd,const MRHumanRestingSupportDispatchGPU& d,
                         id<MTLBuffer> map,NSUInteger offset,id<MTLBuffer> weights,
                         id<MTLBuffer> bodies,id<MTLBuffer> low,id<MTLBuffer> basis,
                         id<MTLBuffer> status,unsigned step) {
        auto reset=[cmd blitCommandEncoder];
        check(reset!=nil,"material endpoint reset encoder");
        [reset fillBuffer:endpointIndex_ range:NSMakeRange(0,sizeof(unsigned)) value:0xff];
        [reset endEncoding];
        auto e=[cmd computeCommandEncoder];
        check(e!=nil,"material endpoint encoder");
        e.label=@"Candidate skin and fat strain domain";
        [e setComputePipelineState:endpoint_];
        [e setBytes:&d length:sizeof(d) atIndex:0];
        [e setBuffer:map offset:offset atIndex:1];
        [e setBuffer:weights offset:0 atIndex:2];
        [e setBuffer:bodies offset:0 atIndex:3];
        [e setBuffer:low offset:0 atIndex:4];
        [e setBuffer:basis offset:0 atIndex:5];
        [e setBuffer:stock_ offset:0 atIndex:6];
        [e setBuffer:endpointIndex_ offset:0 atIndex:7];
        [e setBuffer:fail_ offset:0 atIndex:8];
        [e setBytes:&step length:sizeof(step) atIndex:9];
        [e dispatchThreads:MTLSizeMake(vertices_,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [e endEncoding];
        auto finish=[cmd computeCommandEncoder];
        check(finish!=nil,"material endpoint decision encoder");
        [finish setComputePipelineState:endpointDecision_];
        [finish setBuffer:endpointIndex_ offset:0 atIndex:0];
        [finish setBuffer:status offset:0 atIndex:1];
        [finish dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [finish endEncoding];
    }
