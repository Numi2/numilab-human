#pragma once
#include <CommonCrypto/CommonDigest.h>
#include <fstream>
#include <filesystem>
#include <vector>
#include <array>
#include <iomanip>
#include <sstream>
#include "pelvic_material_assets.hpp"

class NumiHumanPelvicContact {
    id<MTLBuffer> samples=nil,modes=nil,triangles=nil,neighbors=nil,nodes=nil,nodeModes=nil,cells=nil,inverse=nil;
    id<MTLBuffer> coordinates=nil,rows=nil,jacobian=nil,pressures=nil,stats=nil,wrenches=nil,trace=nil,nodePositions=nil,groups=nil,materialTrace=nil,workspace=nil,accepted=nil,needed=nil,history=nil;
    id<MTLComputePipelineState> geometry=nil,solve=nil,wrench=nil,apply=nil,pose=nil,validate=nil,admit=nil,confirm=nil,publisher=nil;
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
        samples=load(0);modes=load(1);triangles=load(2);neighbors=load(3);nodes=load(4);nodeModes=load(5);cells=load(6);inverse=load(7);
        countNodes=unsigned(nodes.length/48);countCells=unsigned(cells.length/16);countGroups=(countCells+127)/128;
        auto alloc=[&](size_t size,bool shared=false){auto b=[dev newBufferWithLength:size options:shared?MTLResourceStorageModeShared:MTLResourceStorageModePrivate];
            if(!b)throw std::runtime_error("pelvic working buffer allocation");return b;};
        accepted=alloc((57+countNodes)*16,true);std::memset(accepted.contents,0,accepted.length);
        needed=alloc(8);history=alloc(std::max(1u,countTrace)*64,true);workspace=alloc(2u*2u*96u*96u*4u);coordinates=alloc(192*4);rows=alloc(512*64);jacobian=alloc(512*96*4);pressures=alloc(512*4);stats=alloc(32);wrenches=alloc(128);
        trace=alloc(std::max(1u,countTrace)*160,true);materialTrace=alloc(std::max(1u,countTrace)*16,true);
        nodePositions=alloc(countNodes*16);groups=alloc(countGroups*16);
        auto pipeline=[&](NSString* name){NSError* err=nil;auto p=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
            if(!p)throw std::runtime_error("missing pelvic material kernel");return p;};
        geometry=pipeline(@"nm_pelvic_contact_rows");solve=pipeline(@"nm_pelvic_contact_solve");wrench=pipeline(@"nm_pelvic_contact_wrench");apply=pipeline(@"nm_pelvic_contact_apply");
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
        auto run=[&](unsigned grids,unsigned width){[e dispatchThreadgroups:MTLSizeMake(grids,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];[e memoryBarrierWithScope:MTLBarrierScopeBuffers];};
        unsigned first=unsigned(p.articulationFirstBody);auto positions=ptr(p.bodyPoses);auto low=ptr(p.bodyPositionLow);
        auto nearest=[&](unsigned displaced,unsigned conditional){
            [e setComputePipelineState:geometry];words(0,{first,displaced,conditional,0});bind(1,samples);bind(2,modes);bind(3,triangles);bind(4,neighbors);
            bind(5,positions);bind(6,low);bind(7,coordinates);bind(8,rows);bind(9,jacobian);bind(10,needed);run(4,128);
        };
        nearest(post,0);
        for(unsigned outer=0;outer<8;++outer){
            [e setComputePipelineState:solve];words(0,{0,outer,384,0});bind(1,samples);bind(2,rows);bind(3,jacobian);bind(4,coordinates);
            bind(5,pressures);bind(6,stats);bind(7,workspace);bind(8,needed);run(2,32);
            nearest(1,outer);
            [e setComputePipelineState:confirm];bind(0,samples);bind(1,rows);bind(2,jacobian);bind(3,coordinates);
            bind(4,pressures);bind(5,stats);words(6,{p.stepIndex,countTrace,outer,unsigned(post)});
            bind(7,trace);bind(8,needed);bind(9,history);run(2,32);
        }
        if(!post){
            [e setComputePipelineState:wrench];words(0,{first,0,0,0});bind(1,samples);bind(2,rows);bind(3,pressures);bind(4,modes);
            bind(5,coordinates);bind(6,positions);bind(7,wrenches);run(4,128);
            [e setComputePipelineState:apply];words(0,{p.stepIndex,unsigned(p.dofCount),unsigned(p.mujocoGeneralizedForceOffset),0});
            words(1,{first,unsigned(p.bodyJacobianPointOffset),countTrace,0});bind(2,wrenches);bind(3,positions);bind(4,ptr(p.pointJacobians));
            bind(5,ptr(p.mujocoGeneralizedForceArena));bind(6,stats);bind(7,trace);run((unsigned(p.dofCount)+63)/64,64);
        }else{
            [e setComputePipelineState:pose];words(0,{first,countNodes,countCells,0});bind(1,nodes);bind(2,nodeModes);bind(3,coordinates);
            bind(4,positions);bind(5,low);bind(6,nodePositions);run((countNodes+127)/128,128);
            [e setComputePipelineState:validate];words(0,{first,countNodes,countCells,0});const std::array<float,4> law{12000,120000,0,0};
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
        [e setBuffer:accepted offset:0 atIndex:8];
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
            const unsigned owners[4]={1,128,131,145};
            for(unsigned i=0;i<4;++i){row("relative_position",owners[i],1+2*i);row("orientation",owners[i],2+2*i);}
            for(unsigned i=0;i<48;++i)row("coordinate_block",i,9+i);
            for(unsigned i=0;i<countNodes;++i)row("material_node",i,57+i);
        }
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
