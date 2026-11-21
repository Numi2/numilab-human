#pragma once
#include <CommonCrypto/CommonDigest.h>
#include <fstream>
#include <filesystem>
#include <vector>
#include <array>
#include <map>
#include <set>
#include <iomanip>
#include <sstream>
#include "pelvic_material_assets.hpp"

class NumiHumanPelvicContact {
    id<MTLBuffer> samples=nil,triangles=nil,nodes=nil,cells=nil,inverse=nil;
    id<MTLBuffer> coordinates=nil,rows=nil,jacobian=nil,pressures=nil,stats=nil,wrenches=nil,trace=nil,nodePositions=nil,groups=nil,materialTrace=nil,workspace=nil,accepted=nil,needed=nil,history=nil;
    id<MTLComputePipelineState> geometry=nil,solve=nil,wrench=nil,apply=nil,pose=nil,validate=nil,admit=nil,confirm=nil,publisher=nil;
    id<MTLBuffer> pool=nil,poolRegions=nil,poolNeighbors=nil,poolGaps=nil,poolSlots=nil,poolMin=nil;
    id<MTLComputePipelineState> queryPool=nil,selectPool=nil,clearPool=nil;
    id<MTLBuffer> flexibility=nil,freeIndex=nil,columns=nil,gram=nil,lever=nil,localN=nil;
    id<MTLComputePipelineState> makeColumns=nil,makeGram=nil,moveNodes=nil;
    id<MTLBuffer> included=nil,proposals=nil,contactCache=nil;
    id<MTLComputePipelineState> exchange=nil;
    id<MTLBuffer> sv=nil,sf=nil,sw=nil,sr=nil,si=nil,sx=nil,sn=nil,vn=nil,boxes=nil,reverse=nil,witness=nil;
    id<MTLComputePipelineState> surfaceVertices=nil,surfaceNormals=nil,vertexNormals=nil,buildBounds=nil,findReverse=nil;
    unsigned numSurface=0,numFaces=0,numReverse=0,numLeaves=0;
    id<MTLBuffer> kr=nil,kc=nil,kv=nil,inc=nil,partial=nil,quality=nil,qualityTrace=nil,failedPre=nil;
    id<MTLComputePipelineState> clearInc=nil,buildInc=nil,measureForce=nil,gate=nil;
    unsigned residualGroups=0;
    id<MTLBuffer> dispatchSizes=nil,dispatchArgs=nil;
    id<MTLComputePipelineState> restoreMaterial=nil,makeSpans=nil,unloaded=nil;
    std::map<unsigned,unsigned> spanIndex;
    unsigned poolCount=0,numRegions=0;
    unsigned countNodes=0,countCells=0,countGroups=0,countTrace=0,encodedSteps=0;
    std::filesystem::path directory;
public:
    void configure(id<MTLDevice> dev,id<MTLLibrary> lib){
        const char* source=std::getenv("NUMI_HUMAN_PELVIC_CONTACT_DATA");
        if(!source||!*source)return;
        directory=source;const char* tr=std::getenv("NUMI_HUMAN_PELVIC_CONTACT_TRACE");
        if(tr&&*tr)countTrace=std::stoul(tr);
        if(countTrace>200000)throw std::runtime_error("pelvic trace exceeds bounded observation");
        auto load=[&](unsigned idx){
            const auto& spec=pelvicMaterialFiles[idx];auto path=directory/spec.name;
            NSData* d=[NSData dataWithContentsOfFile:@(path.c_str())];
            if(d.length!=spec.bytes)throw std::runtime_error("pelvic material payload length: "+path.string());
            unsigned char hash[CC_SHA256_DIGEST_LENGTH];CC_SHA256(d.bytes,static_cast<CC_LONG>(d.length),hash);
            std::ostringstream hex;for(unsigned char b:hash)hex<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(b);
            if(hex.str()!=spec.hash)throw std::runtime_error("pelvic registration identity: "+path.string());
            auto result=[dev newBufferWithBytes:d.bytes length:d.length options:MTLResourceStorageModeShared];
            if(!result)throw std::runtime_error("pelvic device allocation failed");return result;
        };
        triangles=load(2);nodes=load(4);cells=load(6);inverse=load(7);freeIndex=load(11);flexibility=load(12);
        sv=load(13);sf=load(14);sw=load(15);sr=load(16);si=load(17);kr=load(18);kc=load(19);kv=load(20);pool=load(8);poolNeighbors=load(9);poolRegions=load(10);poolCount=unsigned(pool.length/112);numRegions=unsigned(poolRegions.length/16);
        countNodes=unsigned(nodes.length/48);countCells=unsigned(cells.length/16);countGroups=(countCells+127)/128;
        auto alloc=[&](size_t size,bool shared=false){auto b=[dev newBufferWithLength:size options:shared?MTLResourceStorageModeShared:MTLResourceStorageModePrivate];
            if(!b)throw std::runtime_error("pelvic working buffer allocation");return b;};
        samples=alloc(512*80);columns=alloc(countNodes*256u*16u);gram=alloc(2u*65536u*4u);lever=alloc(512u*16u);localN=alloc(512u*16u);residualGroups=(countNodes/2+127)/128;inc=alloc(countNodes*8*4);partial=alloc(2*residualGroups*16);quality=alloc(32);qualityTrace=alloc(std::max(1u,countTrace)*64,true);failedPre=alloc(4);numSurface=unsigned(sv.length/32);numFaces=unsigned(sf.length/48);numReverse=pelvicUniqueVertices;numLeaves=pelvicLeafCount;
        sx=alloc(numSurface*16);sn=alloc(numFaces*16);vn=alloc(numSurface*16);boxes=alloc(4*numLeaves*32);reverse=alloc(numReverse*16);witness=alloc(numReverse*4);included=alloc(poolCount*4);proposals=alloc(numRegions*4);contactCache=alloc(513*4);accepted=alloc((9+2*countNodes+2048)*16,true);std::memset(accepted.contents,0,accepted.length);
        poolMin=alloc(numRegions*4);poolSlots=alloc(512*4);poolGaps=alloc(poolCount*4);needed=alloc(8);history=alloc(std::max(1u,countTrace)*64,true);workspace=alloc(2u*65536u*4u);coordinates=alloc(countNodes*16u);rows=alloc(512*64);jacobian=alloc(512*96*4);pressures=alloc(512*4);stats=alloc(32);wrenches=alloc(128);
        trace=alloc(std::max(1u,countTrace)*160,true);materialTrace=alloc(std::max(1u,countTrace)*16,true);
        nodePositions=alloc(countNodes*16);groups=alloc(countGroups*16);
        auto pipeline=[&](NSString* name){NSError* err=nil;auto p=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
            if(!p)throw std::runtime_error("missing pelvic material kernel");return p;};
        geometry=pipeline(@"nm_pelvic_contact_rows");solve=pipeline(@"nm_pelvic_schur_solve");makeColumns=pipeline(@"nm_pelvic_schur_columns");makeGram=pipeline(@"nm_pelvic_schur_gram");moveNodes=pipeline(@"nm_pelvic_schur_displacement");wrench=pipeline(@"nm_pelvic_contact_wrench");apply=pipeline(@"nm_pelvic_contact_apply");
        std::set<unsigned> gridSizes{1,2,4,16,1024,numRegions,(countNodes+127)/128,(poolCount-numReverse+127)/128,
            (numSurface+127)/128,(numFaces+127)/128,(numReverse+127)/128,(countNodes/2*256+127)/128,(countNodes*8+127)/128,2*residualGroups};
        for(unsigned level=numLeaves;level;level>>=1)gridSizes.insert((2*level+63)/64);
        std::vector<unsigned> sizes(gridSizes.begin(),gridSizes.end());for(unsigned i=0;i<sizes.size();++i)spanIndex.emplace(sizes[i],i);
        dispatchSizes=[dev newBufferWithBytes:sizes.data() length:sizes.size()*4 options:MTLResourceStorageModeShared];
        dispatchArgs=alloc(sizes.size()*12);
        unloaded=pipeline(@"nm_pelvic_unloaded");restoreMaterial=pipeline(@"nm_pelvic_restore_material");makeSpans=pipeline(@"nm_pelvic_optional_spans");
        clearInc=pipeline(@"nm_pelvic_residual_clear");buildInc=pipeline(@"nm_pelvic_residual_incidence");measureForce=pipeline(@"nm_pelvic_free_residual");gate=pipeline(@"nm_pelvic_force_gate");surfaceVertices=pipeline(@"nm_pelvic_surface_vertices");surfaceNormals=pipeline(@"nm_pelvic_surface_normals");
        vertexNormals=pipeline(@"nm_pelvic_surface_vertex_normals");buildBounds=pipeline(@"nm_pelvic_surface_bounds");findReverse=pipeline(@"nm_pelvic_reverse_query");exchange=pipeline(@"nm_pelvic_pool_exchange");queryPool=pipeline(@"nm_pelvic_pool_query");selectPool=pipeline(@"nm_pelvic_pool_select");clearPool=pipeline(@"nm_pelvic_pool_clear");
        publisher=pipeline(@"nm_pelvic_material_publish");confirm=pipeline(@"nm_pelvic_contact_confirm");pose=pipeline(@"nm_pelvic_material_pose");validate=pipeline(@"nm_pelvic_material_validate");admit=pipeline(@"nm_pelvic_material_admit");
    }
    void encode(const metalrobo::MetalNumanXTransactionPass& p,bool post){
        if(!geometry)return;
        if(p.environmentCount!=1||p.bodyCount<=145||p.articulationFirstBody>1||!p.bodyPositionLow||!p.pointJacobians||
           p.bodyJacobianPointOffset+4*p.bodyCount>p.pointCount||p.dofCount==0||p.dofCount>UINT32_MAX)
            throw std::runtime_error("pelvic contact requires registered body poses and point Jacobians");
        auto cb=(__bridge id<MTLCommandBuffer>)p.commandBuffer;auto e=[cb computeCommandEncoder];
        if(!e)throw std::runtime_error("missing pelvic compute encoder");
        auto bind=[&](unsigned index,id<MTLBuffer> b){[e setBuffer:b offset:0 atIndex:index];};
        auto ptr=[](void* v){return (__bridge id<MTLBuffer>)v;};
        using U=std::array<unsigned,4>;auto words=[&](unsigned at,U v){[e setBytes:v.data() length:16 atIndex:at];};
        bool optional=false;
        auto run=[&](unsigned grids,unsigned width){
            auto size=MTLSizeMake(width,1,1);
            if(optional)[e dispatchThreadgroupsWithIndirectBuffer:dispatchArgs indirectBufferOffset:12u*spanIndex.at(grids) threadsPerThreadgroup:size];
            else [e dispatchThreadgroups:MTLSizeMake(grids,1,1) threadsPerThreadgroup:size];[e memoryBarrierWithScope:MTLBarrierScopeBuffers];};
        unsigned first=unsigned(p.articulationFirstBody);auto positions=ptr(p.bodyPoses);auto low=ptr(p.bodyPositionLow);
        auto nearest=[&](unsigned displaced,unsigned conditional){
            [e setComputePipelineState:geometry];words(0,{first,displaced,conditional,0});bind(1,samples);bind(2,pool);bind(3,triangles);bind(4,poolNeighbors);
            bind(5,positions);bind(6,low);bind(7,coordinates);bind(8,rows);bind(9,localN);bind(10,needed);bind(11,poolSlots);bind(12,nodePositions);bind(13,lever);bind(14,reverse);bind(15,freeIndex);run(4,128);
        };
        auto observe=[&](unsigned conditional,unsigned displaced){
            [e setComputePipelineState:pose];words(0,{first,countNodes,countCells,displaced});bind(1,nodes);bind(2,coordinates);bind(3,coordinates);
            bind(4,positions);bind(5,low);bind(6,nodePositions);run((countNodes+127)/128,128);
            [e setComputePipelineState:queryPool];words(0,{first,poolCount,conditional,0});bind(1,pool);bind(2,nodePositions);
            bind(3,positions);bind(4,low);bind(5,triangles);bind(6,poolNeighbors);bind(7,needed);bind(8,poolGaps);run((poolCount-numReverse+127)/128,128);
            [e setComputePipelineState:surfaceVertices];words(0,{first,numSurface,conditional,0});bind(1,sv);bind(2,nodePositions);bind(3,sx);bind(4,needed);run((numSurface+127)/128,128);
            [e setComputePipelineState:surfaceNormals];words(0,{first,numFaces,conditional,0});bind(1,sf);bind(2,sx);bind(3,sn);bind(4,needed);run((numFaces+127)/128,128);
            [e setComputePipelineState:vertexNormals];words(0,{first,numSurface,conditional,0});bind(1,sr);bind(2,si);bind(3,sf);bind(4,sx);bind(5,sn);bind(6,vn);bind(7,needed);run((numSurface+127)/128,128);
            [e setComputePipelineState:buildBounds];bind(1,sf);bind(2,sx);bind(3,boxes);bind(4,needed);
            for(unsigned n=numLeaves;n;n>>=1){words(0,{first,n,conditional,0});run((2*n+63)/64,64);}
            [e setComputePipelineState:findReverse];words(0,{first,poolCount,conditional,0});bind(1,pool);bind(2,positions);bind(3,low);bind(4,nodes);bind(5,sf);bind(6,sx);bind(7,sn);bind(8,vn);bind(9,boxes);bind(10,sw);bind(11,poolGaps);bind(12,reverse);bind(13,needed);bind(14,witness);run((numReverse+127)/128,128);
        };
        auto select=[&](unsigned conditional,bool readOnly){
            [e setComputePipelineState:selectPool];words(0,{0,conditional,unsigned(readOnly),0});bind(1,poolRegions);bind(2,poolGaps);
            bind(3,included);bind(4,proposals);bind(5,poolMin);bind(6,needed);run(numRegions,32);
            if(!readOnly){
                [e setComputePipelineState:exchange];words(0,{numRegions,conditional,0,0});bind(1,poolRegions);bind(2,proposals);bind(3,poolSlots);
                bind(4,included);bind(5,pressures);bind(6,poolGaps);bind(7,needed);run(2,32);
            }
        };
        if(!post){
            [e setComputePipelineState:restoreMaterial];words(0,{p.stepIndex,countNodes,0,0});bind(1,accepted);bind(2,coordinates);run((countNodes+127)/128,128);
            [e setComputePipelineState:clearPool];words(0,{p.stepIndex,poolCount,0,0});bind(1,poolSlots);bind(2,included);bind(3,contactCache);bind(4,pressures);run((poolCount+127)/128,128);
            words(0,{p.stepIndex,poolCount,1,0});run(4,128);
        }
        observe(0,1);select(0,true);
        [e setComputePipelineState:unloaded];words(0,{p.stepIndex,countTrace,numRegions,unsigned(post)});bind(1,coordinates);bind(2,poolMin);
        bind(3,poolRegions);bind(4,needed);bind(5,stats);bind(6,quality);bind(7,pressures);bind(8,trace);bind(9,history);bind(10,qualityTrace);run(2,32);
        for(unsigned outer=0;outer<16;++outer){
            optional=false;
            {unsigned count=unsigned(spanIndex.size());[e setComputePipelineState:makeSpans];[e setBytes:&count length:4 atIndex:0];
                bind(1,needed);bind(2,dispatchSizes);bind(3,dispatchArgs);run((count+31)/32,32);optional=true;}

            select(1,false);
            nearest(1,1);
            for(unsigned side=0;side<2;++side){
                [e setComputePipelineState:makeColumns];words(0,{side,1,0,0});bind(1,poolSlots);bind(2,pool);bind(3,localN);
                bind(4,freeIndex);bind(5,flexibility);bind(6,columns);bind(7,needed);run((countNodes/2*256+127)/128,128);
            }
            [e setComputePipelineState:makeGram];words(0,{0,1,0,0});bind(1,poolSlots);bind(2,pool);bind(3,localN);
            bind(4,columns);bind(5,gram);bind(6,needed);run(1024,128);
            [e setComputePipelineState:solve];words(0,{0,1,0,0});bind(1,poolSlots);bind(2,rows);bind(3,gram);
            bind(4,pressures);bind(5,stats);bind(6,workspace);bind(7,needed);run(2,32);
            [e setComputePipelineState:moveNodes];words(0,{0,1,countNodes,0});bind(1,pressures);bind(2,columns);
            bind(3,coordinates);bind(4,needed);run((countNodes+127)/128,128);
            observe(1,1);select(1,true);nearest(1,1);
            [e setComputePipelineState:clearInc];[e setBytes:&countNodes length:4 atIndex:0];bind(1,inc);run((countNodes*8+127)/128,128);
            [e setComputePipelineState:buildInc];bind(0,poolSlots);bind(1,pressures);bind(2,pool);bind(3,inc);run(16,128);
            [e setComputePipelineState:measureForce];words(0,{0,residualGroups,1,0});bind(1,kr);bind(2,kc);bind(3,kv);bind(4,coordinates);bind(5,inc);bind(6,poolSlots);bind(7,pressures);bind(8,localN);bind(9,pool);bind(10,partial);bind(11,needed);run(2*residualGroups,128);
            [e setComputePipelineState:confirm];bind(0,samples);bind(1,rows);bind(2,gram);bind(3,coordinates);
            bind(4,pressures);bind(5,stats);words(6,{p.stepIndex,countTrace,outer,unsigned(post)});
            bind(7,trace);bind(8,needed);bind(9,history);bind(10,poolMin);bind(11,poolRegions);[e setBytes:&numRegions length:4 atIndex:12];bind(13,partial);[e setBytes:&residualGroups length:4 atIndex:14];bind(15,quality);bind(16,qualityTrace);run(2,32);
        }
        optional=false;[e setComputePipelineState:gate];words(0,{p.stepIndex,0,0,unsigned(post)});bind(1,stats);bind(2,quality);bind(3,failedPre);bind(4,ptr(p.standStatuses));run(1,1);
        if(!post){
            [e setComputePipelineState:wrench];words(0,{first,0,0,0});bind(1,samples);bind(2,rows);bind(3,pressures);bind(4,lever);
            bind(5,coordinates);bind(6,positions);bind(7,wrenches);run(4,128);
            [e setComputePipelineState:apply];words(0,{p.stepIndex,unsigned(p.dofCount),unsigned(p.mujocoGeneralizedForceOffset),0});
            words(1,{first,unsigned(p.bodyJacobianPointOffset),countTrace,0});bind(2,wrenches);bind(3,positions);bind(4,ptr(p.pointJacobians));
            bind(5,ptr(p.mujocoGeneralizedForceArena));bind(6,stats);bind(7,trace);bind(8,failedPre);run((unsigned(p.dofCount)+63)/64,64);
        }else{
            [e setComputePipelineState:pose];words(0,{first,countNodes,countCells,1});bind(1,nodes);bind(2,coordinates);bind(3,coordinates);
            bind(4,positions);bind(5,low);bind(6,nodePositions);run((countNodes+127)/128,128);
            [e setComputePipelineState:validate];words(0,{first,countNodes,countCells,1});const std::array<float,4> law{12000,120000,0,0};
            [e setBytes:law.data() length:16 atIndex:1];bind(2,cells);bind(3,inverse);bind(4,nodePositions);bind(5,groups);run(countGroups,128);
            [e setComputePipelineState:admit];words(0,{p.stepIndex,countGroups,countTrace,0});bind(1,stats);bind(2,groups);
            bind(3,ptr(p.standStatuses));bind(4,materialTrace);run(1,128);encodedSteps=std::max(encodedSteps,p.stepIndex+1);
        }
        [e endEncoding];
    }
    void publish(const metalrobo::MetalNumanXTransactionPass& p,id<MTLBuffer> matter,id<MTLBuffer> respiration){
        if(!geometry)return;
        auto cb=(__bridge id<MTLCommandBuffer>)p.commandBuffer;auto e=[cb computeCommandEncoder];
        if(!e)throw std::runtime_error("missing accepted pelvic material encoder");
        std::array<unsigned,4> d{p.stepIndex,countNodes,unsigned(p.articulationFirstBody),0};
        [e setComputePipelineState:publisher];[e setBytes:d.data() length:16 atIndex:0];
        [e setBuffer:(__bridge id<MTLBuffer>)p.standStatuses offset:0 atIndex:1];
        [e setBuffer:matter offset:0 atIndex:2];[e setBuffer:respiration offset:0 atIndex:3];
        [e setBuffer:(__bridge id<MTLBuffer>)p.bodyPoses offset:0 atIndex:4];
        [e setBuffer:(__bridge id<MTLBuffer>)p.bodyPositionLow offset:0 atIndex:5];
        [e setBuffer:nodePositions offset:0 atIndex:6];[e setBuffer:coordinates offset:0 atIndex:7];
        [e setBuffer:accepted offset:0 atIndex:8];[e setBuffer:poolSlots offset:0 atIndex:9];
        [e setBuffer:pressures offset:0 atIndex:10];[e setBuffer:localN offset:0 atIndex:11];[e setBuffer:contactCache offset:0 atIndex:12];[e setBuffer:pool offset:0 atIndex:13];[e setBuffer:witness offset:0 atIndex:14];
        [e dispatchThreads:MTLSizeMake(countNodes,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];
    }
    ~NumiHumanPelvicContact(){
        const char* file=std::getenv("NUMI_HUMAN_PELVIC_CONTACT_TRIAL_LOG");if(!geometry||!file||!countTrace)return;
        const float* saved=(const float*)accepted.contents;
        if(saved[0]>0&&unsigned(saved[1])==countNodes){
            auto path=std::filesystem::path(file).parent_path()/"accepted-pelvic-state.csv";std::ofstream snap(path);
            snap<<std::setprecision(9)<<"accepted_step,field,index,x,y,z,w\n";
            auto row=[&](const char* field,unsigned index,unsigned at){
                snap<<unsigned(saved[0])<<','<<field<<','<<index;for(unsigned k=0;k<4;++k)snap<<','<<saved[4*at+k];snap<<'\n';};
            for(unsigned i=0;i<512;++i){
                unsigned at=9+2*countNodes+2*i;float point=saved[4*(at+1)];
                if(point>=0){row("normal_and_force",unsigned(point),at);if(unsigned(point)>=poolCount-numReverse){
                    row("reverse_shape",unsigned(point),9+2*countNodes+1024+2*i);
                    row("reverse_parent_nodes",unsigned(point),9+2*countNodes+1025+2*i);
                }}
            }
            const unsigned owners[4]={1,128,131,145};
            for(unsigned i=0;i<4;++i){row("relative_position",owners[i],1+2*i);row("orientation",owners[i],2+2*i);}
            for(unsigned i=0;i<countNodes;++i){row("material_node",i,9+i);row("node_displacement",i,9+countNodes+i);}
        }
        std::ofstream physical(std::filesystem::path(file).parent_path()/"fresh-force-certificate.csv");
        physical<<std::setprecision(9)<<"candidate_step,endpoint,side,free_residual_N,external_force_N,release_gap_m,penetration_m\n";
        const float* z=(const float*)qualityTrace.contents;
        for(unsigned i=0;i<std::min(encodedSteps,countTrace);++i)for(unsigned s=0;s<2;++s)for(unsigned b=0;b<2;++b){
            physical<<i<<','<<s<<','<<b;for(unsigned k=0;k<4;++k)physical<<','<<z[16*i+8*s+4*b+k];physical<<'\n';}
        std::ofstream rounds(std::filesystem::path(file).parent_path()/"contact-convergence.csv");
        rounds<<std::setprecision(9)<<"candidate_step,endpoint,side,refreshes,linear_residual_m,actual_penetration_m,complementarity_J\n";
        const float* h=(const float*)history.contents;
        for(unsigned i=0;i<std::min(encodedSteps,countTrace);++i)for(unsigned stage=0;stage<2;++stage)for(unsigned side=0;side<2;++side){
            unsigned offset=16*i+stage*8+side*4;rounds<<i<<','<<stage<<','<<side;
            for(unsigned j=0;j<4;++j)rounds<<','<<h[offset+j];rounds<<'\n';
        }
        std::ofstream out(file);out<<"candidate_step,gap_r_m,force_r_n,energy_r_j,complementarity_r_j,gap_l_m,force_l_n,energy_l_j,complementarity_l_j,min_J,max_J,stress_Pa,energy_density_J_m3,net_fx_N,net_fy_N,net_fz_N\n";
        const float* d=(const float*)trace.contents;const float* m=(const float*)materialTrace.contents;
        for(unsigned j=0;j<std::min(encodedSteps,countTrace);++j){out<<std::setprecision(9)<<j;
            for(unsigned k=0;k<8;++k)out<<','<<d[j*40+k];for(unsigned k=0;k<4;++k)out<<','<<m[j*4+k];
            for(unsigned k=0;k<3;++k){double n=0;for(unsigned b=0;b<4;++b)n+=d[j*40+8+8*b+k];out<<','<<n;}out<<'\n';}
    }
};
