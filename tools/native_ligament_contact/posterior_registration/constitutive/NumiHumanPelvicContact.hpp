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
    id<MTLBuffer> surfaceReference=nil;
    id<MTLBuffer> sv=nil,sf=nil,sw=nil,sr=nil,si=nil,sx=nil,sn=nil,vn=nil,boxes=nil,reverse=nil,witness=nil;
    id<MTLComputePipelineState> surfaceVertices=nil,surfaceNormals=nil,vertexNormals=nil,buildBounds=nil,findReverse=nil;
    unsigned numSurface=0,numFaces=0,numReverse=0,numLeaves=0;
    id<MTLBuffer> kr=nil,kc=nil,kv=nil,inc=nil,partial=nil,quality=nil,qualityTrace=nil,failedPre=nil;
    id<MTLComputePipelineState> clearInc=nil,buildInc=nil,measureForce=nil,gate=nil;
    id<MTLBuffer> projRows=nil,projCols=nil,projVals=nil;
    id<MTLBuffer> refNodes=nil,refInverse=nil,refPoses=nil,totalDelta=nil,materialState=nil;
    id<MTLBuffer> iterationTrace=nil;
    id<MTLBuffer> lostSupport=nil,priorPressure=nil,forwardMapping=nil,forwardFaces=nil,forwardTriangles=nil,forwardHit=nil;
    id<MTLComputePipelineState> relaxPressure=nil,transformForward=nil;
    id<MTLBuffer> sharedFaces=nil,facetPairs=nil,facetPoses=nil;
    id<MTLComputePipelineState> finiteSurface=nil,candidatePressure=nil;
    unsigned finiteCount=0,reverseStart=0,finiteStart=0;
    unsigned residualGroups=0;
    id<MTLBuffer> volumes=nil,cellRows=nil,cellCorners=nil,fRows=nil,fCols=nil,fVals=nil,fLevels=nil,fOrder=nil,fPerm=nil,fScale=nil;
    id<MTLBuffer> cellLoad=nil,bulkGrad=nil,bulkRhs=nil,baseShift=nil,attachment=nil;
    id<MTLComputePipelineState> cellForce=nil,gatherBulk=nil,prepareRhs=nil,factorApply=nil,requireBulk=nil;
    id<MTLBuffer> cortexFaces=nil,cortexPositions=nil,corticalFrames=nil,correctionControl=nil,rejectedQueries=nil,crossingGroups=nil,crossingTrace=nil;
    id<MTLComputePipelineState> poseCortex=nil,makeCorticalFrames=nil,requestCheck=nil,resumeSolve=nil,measureRejected=nil,testFaces=nil,admitFaces=nil;
    unsigned cortexFaceCount=0,nCrossingGroups=0;
    id<MTLBuffer> capturedSurfaces=nil;
    id<MTLComputePipelineState> captureSurfaces=nil;
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
        volumes=load(21);cellRows=load(22);cellCorners=load(23);fRows=load(24);fCols=load(25);fVals=load(26);
        fLevels=load(27);fOrder=load(28);fPerm=load(29);fScale=load(30);projRows=load(31);projCols=load(32);projVals=load(33);refNodes=load(34);refInverse=load(35);refPoses=load(36);facetPairs=load(37);facetPoses=load(38);finiteCount=unsigned(facetPairs.length/(17*16));
        if constexpr(pelvicMaterialReferenceSurface)surfaceReference=load(39);
        finiteStart=poolCount-pelvicMaterialFacetRows*finiteCount;reverseStart=finiteStart-pelvicUniqueVertices;
        totalDelta=alloc(countNodes*16);materialState=alloc(countCells*16);
        cellLoad=alloc(countCells*64);bulkGrad=alloc(countNodes*16);bulkRhs=alloc(countNodes*16);baseShift=alloc(countNodes*16);attachment=alloc(countNodes*16);
        iterationTrace=alloc(64*std::max(1u,std::min(4u,countTrace))*16,true);
        std::memset(iterationTrace.contents,0,iterationTrace.length);
        samples=alloc(512*80);columns=alloc(countNodes*256u*16u);gram=alloc(2u*65536u*4u);lever=alloc(512u*16u);localN=alloc(512u*16u);residualGroups=(countNodes/2+127)/128;inc=alloc(countNodes*8*4);partial=alloc(2*residualGroups*16);quality=alloc(32);qualityTrace=alloc(std::max(1u,countTrace)*64,true);failedPre=alloc(4);numSurface=unsigned(sv.length/32);numFaces=unsigned(sf.length/48);numReverse=pelvicUniqueVertices;numLeaves=pelvicLeafCount;
        sx=alloc(numSurface*16);sn=alloc(numFaces*16);vn=alloc(numSurface*16);boxes=alloc(4*numLeaves*32);reverse=alloc(numReverse*16);witness=alloc(numReverse*4);included=alloc(poolCount*4);proposals=alloc(numRegions*4);contactCache=alloc(513*4);accepted=alloc((9+3*countNodes+2048)*16,true);std::memset(accepted.contents,0,accepted.length);
        poolMin=alloc(numRegions*4);poolSlots=alloc(512*4);poolGaps=alloc(poolCount*4);needed=alloc(8);history=alloc(std::max(1u,countTrace)*64,true);workspace=alloc(2u*65536u*4u);coordinates=alloc(countNodes*16u);rows=alloc(512*64);jacobian=alloc(512*96*4);pressures=alloc(512*4);stats=alloc(32);wrenches=alloc(128);
        trace=alloc(std::max(1u,countTrace)*160,true);materialTrace=alloc(std::max(1u,countTrace)*16,true);
        nodePositions=alloc(countNodes*16);groups=alloc(countGroups*16);
        auto pipeline=[&](NSString* name){NSError* err=nil;auto p=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
            if(!p)throw std::runtime_error("missing pelvic material kernel");return p;};
        if constexpr(pelvicHasSharedFacets)sharedFaces=load(41);
        if constexpr(pelvicHasSharedForward){lostSupport=alloc(8*4);priorPressure=alloc(512*4);forwardMapping=load(42);forwardFaces=load(43);forwardTriangles=alloc(triangles.length);forwardHit=alloc(forwardMapping.length/2);}
        if constexpr(pelvicHasCrossingCertificate){
            cortexFaces=load(40);cortexFaceCount=unsigned(cortexFaces.length/16);
            nCrossingGroups=(cortexFaceCount+127)/128;
            cortexPositions=alloc(numReverse*16);crossingGroups=alloc(nCrossingGroups*16);
            crossingTrace=alloc(std::max(1u,countTrace)*16,true);
            if(countTrace){rejectedQueries=alloc(6*16,true);std::memset(rejectedQueries.contents,0,rejectedQueries.length);measureRejected=pipeline(@"nm_pelvic_rejected_queries");}
            if constexpr(pelvicHasSharedForward){
                correctionControl=alloc(8*4);
                requestCheck=pipeline(@"nm_pelvic_request_surface_check");resumeSolve=pipeline(@"nm_pelvic_resume_surface_solve");
            }
            corticalFrames=alloc(16*32);makeCorticalFrames=pipeline(@"nm_pelvic_cortical_frames");poseCortex=pipeline(@"nm_pelvic_shared_cortex");
            testFaces=pipeline(@"nm_pelvic_crossing_groups");admitFaces=pipeline(@"nm_pelvic_crossing_admit");
        }
        if constexpr(pelvicHasSharedForward){transformForward=pipeline(@"nm_pelvic_forward_triangles");relaxPressure=pipeline(@"nm_pelvic_pressure_relax");}
        candidatePressure=pipeline(@"nm_pelvic_candidate_pressure");finiteSurface=pipeline(pelvicHasSharedFacets?@"nm_pelvic_finite_native":@"nm_pelvic_finite_surface");cellForce=pipeline(@"nm_pelvic_bulk_corners");gatherBulk=pipeline(@"nm_pelvic_bulk_gather");prepareRhs=pipeline(@"nm_pelvic_bulk_rhs");
        factorApply=pipeline(@"nm_pelvic_bulk_factor");requireBulk=pipeline(@"nm_pelvic_require_bulk");geometry=pipeline(@"nm_pelvic_contact_rows");solve=pipeline(@"nm_pelvic_schur_solve");makeColumns=pipeline(@"nm_pelvic_schur_columns");makeGram=pipeline(@"nm_pelvic_schur_gram");moveNodes=pipeline(@"nm_pelvic_schur_displacement");wrench=pipeline(@"nm_pelvic_contact_wrench");apply=pipeline(@"nm_pelvic_contact_apply");
        std::set<unsigned> gridSizes{1,2,4,16,1024,numRegions,countGroups,(finiteCount+127)/128,(countNodes+127)/128,(reverseStart+127)/128,
            (numSurface+127)/128,(numFaces+127)/128,(numReverse+127)/128,(countNodes/2*256+127)/128,(countNodes*8+127)/128,2*residualGroups};
        for(unsigned level=numLeaves;level;level>>=1)gridSizes.insert((2*level+63)/64);
        std::vector<unsigned> sizes(gridSizes.begin(),gridSizes.end());for(unsigned i=0;i<sizes.size();++i)spanIndex.emplace(sizes[i],i);
        dispatchSizes=[dev newBufferWithBytes:sizes.data() length:sizes.size()*4 options:MTLResourceStorageModeShared];
        dispatchArgs=alloc(sizes.size()*12);
        if(countTrace&&std::getenv("NUMI_HUMAN_PELVIC_CAPTURE_SURFACES")){
            capturedSurfaces=alloc((numSurface+numReverse)*16,true);captureSurfaces=pipeline(@"nm_pelvic_capture_surfaces");
        }
        unloaded=pipeline(@"nm_pelvic_unloaded");restoreMaterial=pipeline(@"nm_pelvic_restore_material");makeSpans=pipeline(@"nm_pelvic_optional_spans");
        clearInc=pipeline(@"nm_pelvic_residual_clear");buildInc=pipeline(@"nm_pelvic_residual_incidence");measureForce=pipeline(@"nm_pelvic_free_residual");gate=pipeline(@"nm_pelvic_force_gate");surfaceVertices=pipeline(pelvicMaterialReferenceSurface?@"nm_pelvic_surface_from_reference":@"nm_pelvic_surface_vertices");surfaceNormals=pipeline(@"nm_pelvic_surface_normals");
        vertexNormals=pipeline(@"nm_pelvic_surface_vertex_normals");buildBounds=pipeline(@"nm_pelvic_surface_bounds");findReverse=pipeline(@"nm_pelvic_reverse_query");exchange=pipeline(@"nm_pelvic_pool_exchange");queryPool=pipeline(pelvicHasSharedForward?@"nm_pelvic_forward_query":@"nm_pelvic_pool_query");selectPool=pipeline(@"nm_pelvic_pool_select");clearPool=pipeline(@"nm_pelvic_pool_clear");
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
        if constexpr(pelvicHasCrossingCertificate){
            [e setComputePipelineState:makeCorticalFrames];[e setBytes:&first length:4 atIndex:0];
            bind(1,positions);bind(2,low);bind(3,corticalFrames);run(1,32);
            [e setComputePipelineState:poseCortex];[e setBytes:&numReverse length:4 atIndex:0];
            bind(1,pool);bind(2,corticalFrames);bind(3,cortexPositions);run((numReverse+127)/128,128);
        }
        if constexpr(pelvicHasSharedForward){
            unsigned n=unsigned(triangles.length/160);
            [e setComputePipelineState:transformForward];[e setBytes:&n length:4 atIndex:0];
            bind(1,forwardFaces);bind(2,cortexPositions);bind(3,triangles);bind(4,corticalFrames);bind(5,forwardTriangles);run((n+127)/128,128);
        }
        if constexpr(pelvicHasSharedForward){
            [e setComputePipelineState:clearInc];unsigned count=1;[e setBytes:&count length:4 atIndex:0];bind(1,lostSupport);run(1,32);
        }
        auto nearest=[&](unsigned displaced,unsigned conditional){
            [e setComputePipelineState:geometry];words(0,{first,displaced,conditional,0});bind(1,samples);bind(2,pool);bind(3,triangles);bind(4,poolNeighbors);
            bind(5,positions);bind(6,low);bind(7,coordinates);bind(8,rows);bind(9,localN);bind(10,needed);bind(11,poolSlots);bind(12,nodePositions);bind(13,lever);bind(14,reverse);bind(15,freeIndex);bind(16,baseShift);bind(17,poolGaps);bind(18,pelvicHasSharedForward?forwardHit:reverse);bind(19,pressures);bind(20,pelvicHasSharedForward?lostSupport:needed);run(4,128);
        };
        auto observe=[&](unsigned conditional,unsigned displaced){
            [e setComputePipelineState:pose];words(0,{first,countNodes,countCells,displaced});bind(1,nodes);bind(2,coordinates);bind(3,coordinates);
            bind(4,positions);bind(5,low);bind(6,nodePositions);bind(7,refNodes);bind(8,refPoses);bind(9,totalDelta);run((countNodes+127)/128,128);
            [e setComputePipelineState:surfaceVertices];words(0,{first,numSurface,conditional,0});bind(1,sv);bind(2,pelvicMaterialReferenceSurface?totalDelta:nodePositions);bind(3,sx);bind(4,needed);
            if constexpr(pelvicMaterialReferenceSurface)bind(5,surfaceReference);run((numSurface+127)/128,128);
            [e setComputePipelineState:queryPool];words(0,{first,poolCount,conditional,0});bind(1,pool);bind(2,pelvicHasSharedForward?sx:nodePositions);
            bind(3,positions);bind(4,low);bind(5,pelvicHasSharedForward?forwardTriangles:triangles);bind(6,poolNeighbors);bind(7,needed);bind(8,poolGaps);
            if constexpr(pelvicHasSharedForward){bind(9,forwardMapping);bind(10,corticalFrames);bind(11,forwardHit);}
            run((reverseStart+127)/128,128);
            [e setComputePipelineState:surfaceNormals];words(0,{first,numFaces,conditional,0});bind(1,sf);bind(2,sx);bind(3,sn);bind(4,needed);run((numFaces+127)/128,128);
            [e setComputePipelineState:vertexNormals];words(0,{first,numSurface,conditional,0});bind(1,sr);bind(2,si);bind(3,sf);bind(4,sx);bind(5,sn);bind(6,vn);bind(7,needed);run((numSurface+127)/128,128);
            [e setComputePipelineState:buildBounds];bind(1,sf);bind(2,sx);bind(3,boxes);bind(4,needed);
            for(unsigned n=numLeaves;n;n>>=1){words(0,{first,n,conditional,0});run((2*n+63)/64,64);}
            [e setComputePipelineState:findReverse];words(0,{first,finiteStart,conditional,0});bind(1,pool);bind(2,positions);bind(3,low);bind(4,nodes);bind(5,sf);bind(6,sx);bind(7,sn);bind(8,vn);bind(9,boxes);bind(10,sw);bind(11,poolGaps);bind(12,reverse);bind(13,needed);bind(14,witness);bind(15,pelvicHasCrossingCertificate?cortexPositions:sx);run((numReverse+127)/128,128);
            [e setComputePipelineState:finiteSurface];words(0,{first,finiteCount,conditional,0});bind(1,facetPairs);bind(2,triangles);bind(3,totalDelta);
            bind(4,positions);bind(5,low);bind(6,facetPoses);bind(7,pool);bind(8,poolGaps);bind(9,needed);
            if constexpr(pelvicHasSharedFacets){bind(10,sharedFaces);bind(11,cortexFaces);bind(12,cortexPositions);bind(13,sx);}
            run((finiteCount+127)/128,128);
        };
        auto select=[&](unsigned conditional,bool readOnly){
            [e setComputePipelineState:selectPool];words(0,{0,conditional,unsigned(readOnly),0});bind(1,poolRegions);bind(2,poolGaps);
            bind(3,included);bind(4,proposals);bind(5,poolMin);bind(6,needed);bind(7,pool);run(numRegions,32);
            if(!readOnly){
                [e setComputePipelineState:exchange];words(0,{numRegions,conditional,0,0});bind(1,poolRegions);bind(2,proposals);bind(3,poolSlots);
                bind(4,included);bind(5,pressures);bind(6,poolGaps);bind(7,needed);run(2,32);
            }
        };
        if(!post){
            [e setComputePipelineState:restoreMaterial];words(0,{p.stepIndex,countNodes,0,0});bind(1,accepted);bind(2,coordinates);run((countNodes+127)/128,128);
            [e setComputePipelineState:clearPool];words(0,{p.stepIndex,poolCount,0,0});bind(1,poolSlots);bind(2,included);bind(3,contactCache);bind(4,pressures);bind(5,accepted);run((poolCount+127)/128,128);
            words(0,{p.stepIndex,poolCount,1,countNodes});run(4,128);
        }
        observe(0,1);select(0,true);
        [e setComputePipelineState:requireBulk];bind(0,needed);run(1,32);
        auto bulkForces=[&](){
            [e setComputePipelineState:cellForce];[e setBytes:&countCells length:4 atIndex:0];bind(1,cells);bind(2,refInverse);bind(3,totalDelta);bind(4,volumes);bind(5,cellLoad);bind(6,materialState);run(countGroups,128);
            [e setComputePipelineState:gatherBulk];[e setBytes:&countNodes length:4 atIndex:0];bind(1,cellRows);bind(2,cellCorners);bind(3,cellLoad);bind(4,bulkGrad);run((countNodes+127)/128,128);
        };
        auto compliance=[&](){
            for(unsigned side=0;side<2;++side){
                [e setComputePipelineState:makeColumns];words(0,{side,1,0,0});bind(1,poolSlots);bind(2,pool);bind(3,localN);
                bind(4,freeIndex);bind(5,flexibility);bind(6,columns);bind(7,needed);run((countNodes/2*256+127)/128,128);
            }
            [e setComputePipelineState:makeGram];words(0,{0,1,0,0});bind(1,poolSlots);bind(2,pool);bind(3,localN);
            bind(4,columns);bind(5,gram);bind(6,needed);run(1024,128);
        };
        auto certify=[&](unsigned outer){
            [e setComputePipelineState:clearInc];[e setBytes:&countNodes length:4 atIndex:0];bind(1,inc);run((countNodes*8+127)/128,128);
            [e setComputePipelineState:buildInc];bind(0,poolSlots);bind(1,pressures);bind(2,pool);bind(3,inc);run(16,128);
            [e setComputePipelineState:measureForce];words(0,{0,residualGroups,1,0});bind(1,kr);bind(2,kc);bind(3,kv);bind(4,coordinates);bind(5,inc);bind(6,poolSlots);bind(7,pressures);bind(8,localN);bind(9,pool);bind(10,partial);bind(11,needed);bind(12,bulkGrad);bind(13,attachment);run(2*residualGroups,128);
            [e setComputePipelineState:confirm];bind(0,samples);bind(1,rows);bind(2,gram);bind(3,coordinates);
            bind(4,pressures);bind(5,stats);words(6,{p.stepIndex,countTrace,outer,unsigned(post)});
            bind(7,trace);bind(8,needed);bind(9,history);bind(10,poolMin);bind(11,poolRegions);[e setBytes:&numRegions length:4 atIndex:12];bind(13,partial);[e setBytes:&residualGroups length:4 atIndex:14];bind(15,quality);bind(16,qualityTrace);bind(17,iterationTrace);run(2,32);
            if(post&&pelvicHasSharedForward){
                bool priorOptional=optional;optional=false;
                [e setComputePipelineState:requestCheck];words(0,{nCrossingGroups,unsigned(outer==UINT_MAX),0,0});
                bind(1,needed);bind(2,correctionControl);run(1,1);
                [e setComputePipelineState:testFaces];[e setBytes:&cortexFaceCount length:4 atIndex:0];
                bind(1,cortexFaces);bind(2,cortexPositions);bind(3,sf);bind(4,sx);bind(5,boxes);bind(6,crossingGroups);
                [e dispatchThreadgroupsWithIndirectBuffer:correctionControl indirectBufferOffset:8 threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [e setComputePipelineState:resumeSolve];words(0,{nCrossingGroups,0,0,0});bind(1,needed);bind(2,correctionControl);
                bind(3,crossingGroups);bind(4,cortexFaces);run(1,128);
                optional=priorOptional;
            }
        };
        bulkForces();
        // A pressure is only an initial guess. Recompute the moving rows, full
        // residual, contact extrema, complementarity and compliance before using it.
        if(p.stepIndex || post){
            nearest(0,0);
            compliance();
            [e setComputePipelineState:candidatePressure];words(0,{0,0,0,0});bind(1,poolSlots);bind(2,rows);
            bind(3,gram);bind(4,pressures);bind(5,stats);run(2,32);
            certify(UINT_MAX);
        }
        for(unsigned outer=0;outer<16;++outer){
            optional=false;
            {unsigned count=unsigned(spanIndex.size());[e setComputePipelineState:makeSpans];[e setBytes:&count length:4 atIndex:0];
                bind(1,needed);bind(2,dispatchSizes);bind(3,dispatchArgs);run((count+31)/32,32);optional=true;}

            [e setComputePipelineState:prepareRhs];[e setBytes:&countNodes length:4 atIndex:0];bind(1,kr);bind(2,kc);bind(3,kv);bind(4,coordinates);bind(5,bulkGrad);bind(6,bulkRhs);bind(7,baseShift);run((countNodes+127)/128,128);
            [e setComputePipelineState:factorApply];bind(0,fRows);bind(1,fCols);bind(2,fVals);bind(3,fLevels);bind(4,fOrder);bind(5,fPerm);bind(6,fScale);bind(7,bulkRhs);bind(8,baseShift);bind(9,needed);bind(10,projRows);bind(11,projCols);bind(12,projVals);run(2,128);
            select(1,false);
            nearest(1,1);
            compliance();
            [e setComputePipelineState:solve];words(0,{0,1,0,0});bind(1,poolSlots);bind(2,rows);bind(3,gram);
            bind(4,pressures);bind(5,stats);bind(6,workspace);bind(7,needed);bind(8,pelvicHasSharedForward?priorPressure:pressures);run(2,32);
            [e setComputePipelineState:moveNodes];words(0,{0,1,countNodes,0});bind(1,pressures);bind(2,columns);
            bind(3,coordinates);bind(4,needed);bind(5,baseShift);bind(6,pelvicHasSharedForward?lostSupport:needed);run((countNodes+127)/128,128);
            if constexpr(pelvicHasSharedForward){
                [e setComputePipelineState:relaxPressure];bind(0,pressures);bind(1,priorPressure);bind(2,lostSupport);bind(3,needed);run(4,128);
            }
            observe(1,1);bulkForces();select(1,true);nearest(1,1);
            certify(outer);
        }
        optional=false;[e setComputePipelineState:gate];words(0,{p.stepIndex,0,0,unsigned(post)});bind(1,stats);bind(2,quality);bind(3,failedPre);bind(4,ptr(p.standStatuses));run(1,1);
        if(!post){
            [e setComputePipelineState:wrench];words(0,{first,countNodes,0,0});bind(1,samples);bind(2,rows);bind(3,pressures);bind(4,lever);
            bind(5,coordinates);bind(6,positions);bind(7,wrenches);bind(8,nodes);bind(9,attachment);run(4,128);
            [e setComputePipelineState:apply];words(0,{p.stepIndex,unsigned(p.dofCount),unsigned(p.mujocoGeneralizedForceOffset),0});
            words(1,{first,unsigned(p.bodyJacobianPointOffset),countTrace,0});bind(2,wrenches);bind(3,positions);bind(4,ptr(p.pointJacobians));
            bind(5,ptr(p.mujocoGeneralizedForceArena));bind(6,stats);bind(7,trace);bind(8,failedPre);run((unsigned(p.dofCount)+63)/64,64);
        }else{
            [e setComputePipelineState:pose];words(0,{first,countNodes,countCells,1});bind(1,nodes);bind(2,coordinates);bind(3,coordinates);
            bind(4,positions);bind(5,low);bind(6,nodePositions);bind(7,refNodes);bind(8,refPoses);bind(9,totalDelta);run((countNodes+127)/128,128);
            [e setComputePipelineState:validate];words(0,{first,countNodes,countCells,1});const std::array<float,4> law{12000,120000,0,0};
            [e setBytes:law.data() length:16 atIndex:1];bind(2,cells);bind(3,inverse);bind(4,nodePositions);bind(5,groups);bind(6,materialState);run(countGroups,128);
            [e setComputePipelineState:admit];words(0,{p.stepIndex,countGroups,countTrace,0});bind(1,stats);bind(2,groups);
            bind(3,ptr(p.standStatuses));bind(4,materialTrace);run(1,128);
            if constexpr(pelvicHasCrossingCertificate){

                [e setComputePipelineState:testFaces];[e setBytes:&cortexFaceCount length:4 atIndex:0];
                bind(1,cortexFaces);bind(2,cortexPositions);bind(3,sf);bind(4,sx);bind(5,boxes);bind(6,crossingGroups);run(nCrossingGroups,128);
                [e setComputePipelineState:admitFaces];words(0,{p.stepIndex,nCrossingGroups,countTrace,numSurface});
                bind(1,crossingGroups);bind(2,sx);bind(3,ptr(p.standStatuses));bind(4,crossingTrace);run(1,128);
                if(countTrace){[e setComputePipelineState:measureRejected];words(0,{p.stepIndex,countTrace,poolCount,0});bind(1,ptr(p.standStatuses));bind(2,poolGaps);bind(3,pool);bind(4,rejectedQueries);run(6,128);}
            }
            encodedSteps=std::max(encodedSteps,p.stepIndex+1);
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
        [e setBuffer:pressures offset:0 atIndex:10];[e setBuffer:localN offset:0 atIndex:11];[e setBuffer:contactCache offset:0 atIndex:12];[e setBuffer:pool offset:0 atIndex:13];[e setBuffer:witness offset:0 atIndex:14];[e setBuffer:totalDelta offset:0 atIndex:15];
        [e dispatchThreads:MTLSizeMake(countNodes,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        if(captureSurfaces){
            d={p.stepIndex,numSurface,numReverse,unsigned(p.articulationFirstBody)};
            [e setComputePipelineState:captureSurfaces];[e setBytes:d.data() length:16 atIndex:0];
            [e setBuffer:pool offset:0 atIndex:6];[e setBuffer:sx offset:0 atIndex:7];[e setBuffer:capturedSurfaces offset:0 atIndex:8];[e setBuffer:(pelvicHasCrossingCertificate?cortexPositions:sx) offset:0 atIndex:9];
            [e dispatchThreads:MTLSizeMake(std::max(numSurface,numReverse),1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        }
        [e endEncoding];
    }
    ~NumiHumanPelvicContact(){
        const char* file=std::getenv("NUMI_HUMAN_PELVIC_CONTACT_TRIAL_LOG");if(!geometry||!file||!countTrace)return;
        if constexpr(pelvicHasCrossingCertificate){
            std::ofstream g(std::filesystem::path(file).parent_path()/"pelvic-endpoint-crossings.csv");
            g<<"candidate_step,transverse_pairs,first_cortical_row,first_material_face,nonfinite\n";
            auto q=static_cast<const unsigned*>(crossingTrace.contents);
            unsigned acceptedCount=unsigned(static_cast<const float*>(accepted.contents)[0]);
            for(unsigned i=0;i<std::min(acceptedCount+1,std::min(encodedSteps,countTrace));++i)
                g<<i<<','<<q[4*i]<<','<<q[4*i+1]<<','<<q[4*i+2]<<','<<q[4*i+3]<<'\n';
        }
        const float* saved=(const float*)accepted.contents;
        if(rejectedQueries&&encodedSteps>unsigned(saved[0])&&unsigned(saved[0])<countTrace){
            unsigned step=unsigned(saved[0]);auto w=static_cast<const unsigned*>(rejectedQueries.contents);
            std::ofstream g(std::filesystem::path(file).parent_path()/"rejected-cortical-queries.csv");
            g<<"candidate_step,side,direction,row,raw_m,cortical_body\n"<<std::setprecision(9);
            for(unsigned j=0;j<6;++j){float gap=0;std::memcpy(&gap,w+4*j+1,4);if(w[4*j+3]==step+1)g<<step<<','<<j/3<<','<<j%3<<','<<w[4*j]<<','<<gap<<','<<w[4*j+2]<<'\n';}
        }
        if(saved[0]>0&&unsigned(saved[1])==countNodes){
            auto path=std::filesystem::path(file).parent_path()/"accepted-pelvic-state.csv";std::ofstream snap(path);
            snap<<std::setprecision(9)<<"accepted_step,field,index,x,y,z,w\n";
            auto row=[&](const char* field,unsigned index,unsigned at){
                snap<<unsigned(saved[0])<<','<<field<<','<<index;for(unsigned k=0;k<4;++k)snap<<','<<saved[4*at+k];snap<<'\n';};
            for(unsigned i=0;i<512;++i){
                unsigned at=9+2*countNodes+2*i;float point=saved[4*(at+1)];
                if(point>=0){row("normal_and_force",unsigned(point),at);if(unsigned(point)>=reverseStart){
                    row("reverse_shape",unsigned(point),9+2*countNodes+1024+2*i);
                    row("reverse_parent_nodes",unsigned(point),9+2*countNodes+1025+2*i);
                }}
            }
            const unsigned owners[4]={1,128,131,145};
            for(unsigned i=0;i<4;++i){row("relative_position",owners[i],1+2*i);row("orientation",owners[i],2+2*i);}
            for(unsigned i=0;i<countNodes;++i){row("material_node",i,9+i);row("node_displacement",i,9+countNodes+i);row("material_delta",i,9+2*countNodes+2048+i);}
        }
        if(capturedSurfaces&&saved[0]>0){
            auto path=std::filesystem::path(file).parent_path()/"accepted-pelvic-surfaces.csv";
            std::ofstream stream(path);stream<<std::setprecision(9)<<"accepted_step,field,index,x,y,z\n";
            const float* coords=(const float*)capturedSurfaces.contents;
            for(unsigned i=0;i<numSurface+numReverse;++i){
                stream<<unsigned(saved[0])<<','<<(i<numSurface?"material_vertex":"cortical_vertex")<<','<<(i<numSurface?i:i-numSurface);
                for(unsigned k=0;k<3;++k)stream<<','<<coords[4*i+k];stream<<'\n';
            }
        }
        std::ofstream iter(std::filesystem::path(file).parent_path()/"bulk-force-iterations.csv");
        iter<<std::setprecision(9)<<"candidate_step,endpoint,iteration,side,residual_N,external_N,release_m,penetration_m\n";
        auto data=static_cast<const float*>(iterationTrace.contents);
        for(unsigned i=0;i<std::min(4u,std::min(encodedSteps,countTrace));++i)for(unsigned e=0;e<2;++e){
            for(unsigned k=0;k<16;++k)for(unsigned side=0;side<2;++side){
                auto q=data+4*(((i*2+e)*16+k)*2+side);
                iter<<i<<','<<e<<','<<k<<','<<side;for(unsigned j=0;j<4;++j)iter<<','<<q[j];iter<<'\n';
            }
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
