        const bool refreshGlides = exactEnvironmentOptIn("NM_HUMAN_GLIDE_REFRESH");
        HumanGlideRefreshTrial glidePlan{};
        if (refreshGlides) {
            const auto rows = state_->femBodyContactSolverSamples;
            if (state_->executionMode != NumiHumanTendonFEMLoadExecutionMode::standaloneRuntime ||
                state_->environmentCount != 1u || state_->compactContactLogicalRows != 0u ||
                state_->hasMultiMasterTriangleContact || state_->implicitSpringCount != 0u ||
                rows.empty() || rows.size() != state_->femBodyContactSamplesCount() ||
                state_->femBodyContactWrenchBuffer == nil || humanGlideRefresh != nullptr) {
                state_->message = "plane refresh requires standalone one-environment plane-only contacts";
                return false;
            }
            if (state_->glideRefreshPipeline == nil) {
                std::vector<unsigned char> seen(state_->nodeLoads.size(), 0u);
                for (const auto& row : rows) {
                    if (row.flags != NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE ||
                        row.slaveNode >= seen.size() || seen[row.slaveNode] != 0u) {
                        state_->message = "plane refresh requires unique unilateral slave rows";
                        return false;
                    }
                    seen[row.slaveNode] = 1u;
                }
                const auto pipeline = [&](NSString* name) {
                    id<MTLFunction> function = [state_->library newFunctionWithName:name];
                    return function != nil
                        ? [state_->device newComputePipelineStateWithFunction:function error:nullptr] : nil;
                };
                state_->glideRefreshPipeline = pipeline(
                    @"numi_matter_metal::nm_human_fascia_refresh_plane_load");
                state_->glideWrenchPipeline = pipeline(
                    @"numi_matter_metal::nm_human_fascia_finish_plane_wrench");
                state_->glidePreviousForce = [state_->device newBufferWithLength:
                    rows.size()*sizeof(nm_float4) options:MTLResourceStorageModePrivate];
                state_->glideRowImpulses = [state_->device newBufferWithLength:
                    rows.size()*sizeof(NMNumiHumanBodyWrenchGPU) options:MTLResourceStorageModePrivate];
                if (state_->glideRefreshPipeline == nil || state_->glideWrenchPipeline == nil ||
                    state_->glidePreviousForce == nil || state_->glideRowImpulses == nil) {
                    state_->glideRefreshPipeline = nil;
                    state_->message = "plane refresh pipeline or impulse scratch unavailable";
                    return false;
                }
                std::fprintf(stderr,"human_fascia_plane_refresh=enabled rows=%zu\n", rows.size());
            }
            glidePlan.dispatch = solverDispatch;
            glidePlan.refresh = state_->glideRefreshPipeline;
            glidePlan.finish = state_->glideWrenchPipeline;
            glidePlan.samples = state_->femBodyContactSolverSampleBuffer;
            glidePlan.poses = bodyPoses;
            glidePlan.previousForce = state_->glidePreviousForce;
            glidePlan.rowImpulses = state_->glideRowImpulses;
            glidePlan.wrenches = state_->femBodyContactWrenchBuffer;
            glidePlan.external = state_->externalForceBuffer;
        }
