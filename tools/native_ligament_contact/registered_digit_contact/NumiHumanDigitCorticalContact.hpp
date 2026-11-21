#pragma once

// Relative material-point rows for frictionless cortical contact in the existing
// articulated solve. The first rows still belong to the external bed provider.
class NumiHumanDigitCorticalContact {
    using Program = metalrobo::MetalNumiHumanSupportGeometryProgram;
    using Pass = metalrobo::MetalNumiHumanSupportGeometryPass;
    Program external{};
    id<MTLDevice> device=nil;
    id<MTLComputePipelineState> pipeline=nil,facePipeline=nil;
    id<MTLBuffer> surface=nil,faceResults=nil;
    std::array<mr_uint4,4> surfaceRanges{};
    unsigned faceStride=0;
    id<MTLBuffer> features=nil,trace=nil,poseTrace=nil;
    unsigned externalRows=0,firstPoint=0,traceSteps=0;
    std::string evidenceDir;
    struct Feature { mr_uint4 bodies; mr_float4 x[4]; };
    std::array<Feature,4> source{};
    static bool encode(void* state,const Pass& p){return static_cast<NumiHumanDigitCorticalContact*>(state)->geometry(p);}
    static void abort(void* state,void* cb){auto& owner=*static_cast<NumiHumanDigitCorticalContact*>(state);owner.external.abort(owner.external.context,cb);}
    bool geometry(const Pass& p){
        if(p.environmentCount!=1 || p.standContactCount!=contacts.size() || p.pointCount!=points.size() ||
           !p.bodyPoses || !p.bodyPositionLow || !p.pointJacobians || !p.pointPositionLow || !p.standContacts)return false;
        Pass bed=p;bed.standContactCount=externalRows;
        if(!external.encodePreDynamics(external.context,bed))return false;
        auto cb=(__bridge id<MTLCommandBuffer>)p.commandBuffer;
        if(!cb || cb.device!=device)return false;
        auto e=[cb computeCommandEncoder];if(!e)return false;
        [e setComputePipelineState:facePipeline];
        mr_uint4 faceSize{p.articulationFirstBody,faceStride,0,0};
        [e setBytes:&faceSize length:sizeof(faceSize) atIndex:0];
        [e setBytes:surfaceRanges.data() length:sizeof(surfaceRanges) atIndex:1];
        [e setBuffer:surface offset:0 atIndex:2];[e setBuffer:features offset:0 atIndex:3];
        [e setBuffer:(__bridge id<MTLBuffer>)p.bodyPoses offset:0 atIndex:4];
        [e setBuffer:(__bridge id<MTLBuffer>)p.bodyPositionLow offset:0 atIndex:5];
        [e setBuffer:faceResults offset:0 atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake(4*faceStride,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        id<MTLResource> dependency=faceResults;[e memoryBarrierWithResources:&dependency count:1];
        [e setComputePipelineState:pipeline];
        [e setBuffer:faceResults offset:0 atIndex:13];
        [e setBytes:&faceStride length:sizeof(faceStride) atIndex:14];
        mr_uint4 d{p.articulationFirstBody,p.dofCount,p.bodyJacobianPointOffset,p.stepIndex};
        mr_uint4 rows{externalRows,firstPoint,unsigned(p.phase==metalrobo::MetalNumiHumanSupportGeometryPhase::predictedFreeEndpoint),traceSteps};
        [e setBytes:&d length:sizeof(d) atIndex:0];[e setBytes:&rows length:sizeof(rows) atIndex:1];[e setBuffer:features offset:0 atIndex:2];
        void* leased[]={p.bodyPoses,p.bodyPositionLow,p.pointWorld,p.pointPositionLow,p.pointJacobians,p.standContacts};
        for(unsigned i=0;i<6;++i)[e setBuffer:(__bridge id<MTLBuffer>)leased[i] offset:0 atIndex:i+3];
        mr_float4 ground=p.groundNormal;ground.w=external.usePerContactSupportPlanes?1.f:0.f;
        [e setBytes:&ground length:sizeof(ground) atIndex:10];[e setBytes:&p.groundPoint length:sizeof(p.groundPoint) atIndex:11];
        [e setBuffer:poseTrace offset:0 atIndex:12];[e setBuffer:trace offset:0 atIndex:9];[e dispatchThreadgroups:MTLSizeMake(source.size(),1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e endEncoding];return true;
    }
public:
    std::vector<MRNumiHumanStandContactGPU> contacts;
    std::vector<MRArticulatedPointImpulseGPU> points;
    static bool requested(){const char* p=std::getenv("NUMI_HUMAN_DIGIT_CORTICAL_FEATURES");return p && *p;}
    Program program() {return {this,&encode,&abort,external.fingerprint^featureFingerprint,external.usePerContactSupportPlanes};}
    std::uint64_t featureFingerprint=1469598103934665603ull;
    void configure(const Program& base,const metalrobo::MetalArticulatedOperatorInput& input,unsigned steps){
        require(base.valid(),"digit contact needs the loaded native bed geometry provider");
        external=base;externalRows=unsigned(input.stand.contacts.size());firstPoint=unsigned(input.points.size());
        contacts.assign(input.stand.contacts.begin(),input.stand.contacts.end());points.assign(input.points.begin(),input.points.end());
        require(contacts.size()+source.size()<=MR_NUMI_HUMAN_STAND_MAX_CONTACTS,"registered internal contact capacity");
        std::ifstream file(std::getenv("NUMI_HUMAN_DIGIT_CORTICAL_FEATURES"),std::ios::binary);
        require(bool(file.read(reinterpret_cast<char*>(source.data()),sizeof(source)))&&file.peek()==std::char_traits<char>::eof(),"incomplete registered digit features");
        for(auto byte:std::as_bytes(std::span(source))){featureFingerprint^=std::to_integer<unsigned char>(byte);featureFingerprint*=1099511628211ull;}
        for(unsigned i=0;i<source.size();++i){
            const auto& f=source[i];auto a=f.bodies.x;auto b=f.bodies.y;
            require(a!=b&&4*a<input.points.size()&&4*b<input.points.size()&&input.points[4*a].bodyIndex==a&&input.points[4*b].bodyIndex==b,"cortical feature body frame mismatch");
            require(f.bodies.z<2&&f.bodies.w<2,"invalid cortical feature type");
            for(auto x:f.x)require(std::isfinite(x.x)&&std::isfinite(x.y)&&std::isfinite(x.z),"nonfinite cortical feature");
            MRNumiHumanStandContactGPU c{};c.pointQueryIndex=unsigned(points.size());c.bodyIndex=a;c.sourceGeometryIndex=0x40000000u+i;
            if(base.usePerContactSupportPlanes)c.planeNormal={0,0,1,0};c.frictionSlopAndStabilization={0,.00025f,1,0};
            points.push_back(input.points[4*a]);contacts.push_back(c);
        }
        device=MTLCreateSystemDefaultDevice();NSError* error=nil;
        auto lib=[device newLibraryWithURL:[NSURL fileURLWithPath:@(NUMI_HUMAN_RESPIRATION_METALLIB)] error:&error];
        auto fn=[lib newFunctionWithName:@"nm_human_digit_cortical_rows"];require(fn!=nil,"missing native digit cortical kernel");
        pipeline=[device newComputePipelineStateWithFunction:fn error:&error];require(pipeline!=nil&&pipeline.threadExecutionWidth==32,"digit cortex pipeline failed");
        auto query=[lib newFunctionWithName:@"nm_digit_surface_faces"];
        facePipeline=[device newComputePipelineStateWithFunction:query error:&error];
        require(facePipeline && facePipeline.threadExecutionWidth==32,"cortical surface query unavailable");
        const auto root=std::filesystem::path(std::getenv("NUMI_HUMAN_DIGIT_CORTICAL_FEATURES")).parent_path();
        std::ifstream geometry(root/"proximal-surface.bin",std::ios::binary|std::ios::ate);
        const auto length=geometry.tellg();
        require(length>std::streamoff(sizeof(surfaceRanges)) && length<4*1024*1024,"invalid cortical surface size");
        geometry.seekg(0);geometry.read(reinterpret_cast<char*>(surfaceRanges.data()),sizeof(surfaceRanges));
        std::vector<unsigned char> payload(std::size_t(length)-sizeof(surfaceRanges));
        require(bool(geometry.read(reinterpret_cast<char*>(payload.data()),payload.size())) && payload.size()%48==0,"invalid native cortical triangle data");
        for(auto bytes:std::as_bytes(std::span(surfaceRanges))){featureFingerprint^=std::to_integer<unsigned char>(bytes);featureFingerprint*=1099511628211ull;}
        for(auto bytes:payload){featureFingerprint^=bytes;featureFingerprint*=1099511628211ull;}
        for(auto range:surfaceRanges){
            require(range.y && range.w && std::uint64_t(range.x)+range.y<=payload.size()/48 &&
                    std::uint64_t(range.z)+range.w<=payload.size()/48,"cortical surface range exceeds its source");
            faceStride=std::max(faceStride,range.y);
        }
        surface=[device newBufferWithBytes:payload.data() length:payload.size() options:MTLResourceStorageModeShared];
        faceResults=[device newBufferWithLength:4*faceStride*48 options:MTLResourceStorageModePrivate];
        require(surface && faceResults,"could not allocate registered cortical surfaces");
        features=[device newBufferWithBytes:source.data() length:sizeof(source) options:MTLResourceStorageModeShared];
        const char* evid=std::getenv("NUMI_HUMAN_DIGIT_CONTACT_TRACE");if(evid&&*evid){evidenceDir=evid;traceSteps=steps;}
        auto bytes=std::max<std::size_t>(16,std::size_t(traceSteps)*8*sizeof(mr_float4));
        trace=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];poseTrace=[device newBufferWithLength:std::max<std::size_t>(16,std::size_t(traceSteps)*12*sizeof(mr_float4)) options:MTLResourceStorageModeShared];
        require(features&&trace&&poseTrace,"could not allocate cortical scatter");
        for(unsigned r=0;r<source.size();r+=2)
            require(source[r].bodies.x==source[r+1].bodies.x && source[r].bodies.y==source[r+1].bodies.y,
                    "digit pose trace requires adjacent registered rows to share a cortical pair");
    }
    void observe(const metalrobo::MetalArticulatedOperatorResult& result,unsigned committed){
        if(evidenceDir.empty() || result.standContactImpulses.empty())return;
        require(result.standContactImpulses.size()==3*contacts.size(),"digit reaction result layout");
        std::ofstream f(evidenceDir+"/digit-contact-impulses.csv",std::ios::app);require(f.good(),"cannot open digit impulse evidence");f.precision(11);
        for(unsigned i=0;i<source.size();++i)f<<committed<<','<<i<<','<<result.standContactImpulses[3*(externalRows+i)]<<'\n';
    }
    template<class T> struct Tail {
        std::vector<T>& src;std::vector<T> extra;
        Tail(std::vector<T>& v,std::size_t count,std::size_t total):src(v){
            if(v.empty())return;
            require(v.size()==total && count<=total,"digit geometry observer shape mismatch");
            extra.assign(v.begin()+count,v.end());v.resize(count);
        }
        ~Tail(){src.insert(src.end(),extra.begin(),extra.end());}
    };
    template<class Observe> void externalObservation(metalrobo::MetalArticulatedOperatorResult& r,unsigned nv,Observe observe){
        // Existing bed and momentum observers consume only external reactions.
        // Preserve the internal impulses and actual point rows on the solver result.
        Tail<float> forces(r.standContactImpulses,3*externalRows,3*contacts.size());
        Tail<mr_float4> normals(r.standContactPlaneNormals,externalRows,contacts.size());
        Tail<MRArticulatedPointWorldGPU> pos(r.pointWorld,firstPoint,points.size());
        Tail<mr_float4> low(r.pointPositionLow,firstPoint,points.size());
        Tail<float> jac(r.pointJacobians,3*nv*firstPoint,3*nv*points.size());
        observe();
    }
    ~NumiHumanDigitCorticalContact(){
        if(evidenceDir.empty()||!trace)return;
        if(poseTrace){
            std::ofstream poses(evidenceDir+"/digit-material-frames.csv");
            poses<<"input_step,body,x,y,z,low_x,low_y,low_z,qx,qy,qz,qw\n";
            poses.precision(10);
            auto v=static_cast<const mr_float4*>(poseTrace.contents);
            for(unsigned step=0;step<traceSteps;++step)for(unsigned j=0;j<4;++j){
                auto t=v+3*(4*step+j);
                if(t[0].w!=1.f)continue;
                auto pair=source[j&~1u].bodies;
                poses<<step<<','<<((j&1u)?pair.y:pair.x);
                for(unsigned ix=0;ix<3;++ix)poses<<','<<t[ix].x<<','<<t[ix].y<<','<<t[ix].z;
                poses<<','<<t[2].w<<'\n';
            }
        }
        std::ofstream f(evidenceDir+"/digit-contact-geometry.csv");f<<"candidate_step,phase,row,gap_m,s,t,u\n";f.precision(11);
        auto p=static_cast<const mr_float4*>(trace.contents);
        for(unsigned k=0;k<traceSteps;++k)for(unsigned phase=0;phase<2;++phase)for(unsigned i=0;i<source.size();++i){
            auto r=p[(k*2+phase)*4+i];if(r.x==0&&r.y==0&&r.z==0&&r.w==0)continue;
            f<<k<<','<<phase<<','<<i<<','<<r.x<<','<<r.y<<','<<r.z<<','<<r.w<<'\n';
        }
    }
};
