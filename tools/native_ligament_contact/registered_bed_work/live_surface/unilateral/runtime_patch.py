"""Install the registered fixed-bed primal/dual row in the existing Newton graph."""
def patch(s, host_lambdas):
    def replace(key, value):
        nonlocal s
        if s.count(key)!=1:
            raise ValueError(f"Unexpected source count {s.count(key)} for {key[:90]}")
        s=s.replace(key,value)
    def before(key,value):
        replace(key,value+"\n"+key)

    before("    bool referenceAreaPlane = false;", """    bool unilateralBed = false;
    id<MTLBuffer> bedDualLinearization = nil;""")
    key="        candidate->acceptedStateProofProgramFingerprint ="
    before(key,'''        if (const char* flag=std::getenv("NM_FEM_REGISTERED_BED_DUAL")) {
            if (std::strcmp(flag,"1") && std::strcmp(flag,"0")) {
                diagnostics.message="invalid registered bed dual option"; return diagnostics;
            }
            candidate->unilateralBed=!std::strcmp(flag,"1");
        }
        if (candidate->unilateralBed) {
            const char* area=std::getenv("NM_FEM_REFERENCE_AREA_PLANE");
            if (!area || std::strcmp(area,"1") || world.contact.pairs.empty() ||
                world.contact.rigidProxies.size()!=1u || world.dispatch.environmentCount!=1u ||
                world.contact.rigidProxies.front().bodyIndex!=NM_INVALID_INDEX ||
                world.contact.rigidProxies.front().adaptiveObjectIndex!=NM_INVALID_INDEX ||
                candidate->fgmresFlatColumnICBEnabled || candidate->fgmresAggregateCycleICBEnabled ||
                candidate->fgmresAggregateFirstCycleICBEnabled) {
                diagnostics.message="registered bed dual requires a single fixed area bed and separate operator and orthogonalization commands";
                return diagnostics;
            }
            candidate->executionFingerprint=mixFingerprint(candidate->executionFingerprint,0x4245444455414c31ull);
        }''')
    before('            "nm_mixed_certify_finish",','''            "nm_fem_registered_bed_residual",
            "nm_fem_registered_bed_operator",
            "nm_fem_registered_bed_certify",''')
    key='        for (const char* name : kernelNames) {'
    replace(key,key+'''
            if (!candidate->unilateralBed && std::string_view(name).starts_with("nm_fem_registered_bed_")) continue;''')
    before("                candidate->planeOffDiagonal=privateScratch",'''                if (candidate->unilateralBed) {
                    candidate->bedDualLinearization=privateScratch<nm_float4>(candidate->device,
                        world.contact.pairs.size(),valid,candidate->residentBytes);
                    std::fprintf(stderr,"registered_bed_dual=1 pairs=%zu\\n",world.contact.pairs.size());
                }''')
    before("        const NSUInteger femHumanAttachmentTotal =\n            environments * state.dispatch.femHumanAttachmentCount;",host_lambdas)
    before("            const char* regionalOnlyCaptureText =",'''            if (state.unilateralBed && femResidualArmijoEnabled) {
                diagnostics.message="registered bed dual requires the physical geometry backtracking path";
                return diagnostics;
            }''')
    before("            if (!encodeMPMNewtonRHSRowsTrace()) {","            encodeRegisteredBedResidual(0u);")
    key="                return true;\n            };\n\n            const bool useAcceptedSubspacePredictor"
    before(key,'''                encodeRegisteredBedOperator(arnoldiOperatorInput,arnoldiOperatorInputOffset,
                    operatorOutput,operatorStates,forceComplete ? 1u : 0u);''')
    key="                if (state.femHuman6SchurCondensedFGMRESEnabled) {\n                    if (!encodeSchurSolve(state.femOperatorValue)"
    before(key,'''                if (state.fgmresOperatorICBEnabled) {
                    encodeRegisteredBedOperator(arnoldiOperatorInput,arnoldiOperatorInputOffset,
                        state.femOperatorValue,state.fgmresStates,0u);
                }''')
    before('            dispatchThreads("nm_contact_reduce_rigid", proxyTotal, [&] {','            encodeRegisteredBedResidual(1u);')
    before('            if (!encodeFEMNewtonCertificateTrace()) {','''            if (state.unilateralBed) {
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                dispatchThreads("nm_fem_registered_bed_certify",pairTotal,[&] {
                    setDispatch(); bindPrimalContactArguments(1u);
                    [encoder setBuffer:state.bedDualLinearization offset:0u atIndex:2u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:3u];
                });
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }''')
    before('            "nm_fem_registered_bed_residual",','            "nm_fem_registered_bed_direction",')
    before('                const std::uint32_t collapseCompletedBacktracking =','''                if (state.unilateralBed) {
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    dispatchThreads("nm_fem_registered_bed_direction",pairTotal,[&] {
                        setDispatch(); bindPrimalContactArguments(1u);
                        [encoder setBuffer:state.femCandidate offset:0u atIndex:2u];
                        [encoder setBuffer:state.bedDualLinearization offset:0u atIndex:3u];
                        [encoder setBuffer:state.femSolution offset:0u atIndex:4u];
                        [encoder setBuffer:state.schedulers offset:0u atIndex:5u];
                        [encoder setBuffer:state.mixedSolver offset:0u atIndex:6u];
                        [encoder setBuffer:state.fgmresStates offset:0u atIndex:7u];
                    });
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }''')
    return s
