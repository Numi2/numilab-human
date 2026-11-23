#pragma once
#include <map>
#include <memory>
#include <utility>

// Setup is host-only. A source column stores its two physical thicknesses and
// projected area. Simulation reads only GPU height keys, rigid point Jacobians
// and the immutable anatomical bindings.
class NumiHumanResolvedBed {
    struct Term { mr_float4 local; mr_uint4 index; };
    using Pair=std::array<unsigned,2>;
    id<MTLDevice> device_;
    id<MTLComputePipelineState> columns_,tiles_,donors_,rows_;
    id<MTLBuffer> stock_,terms_,tilesMap_,donorsMap_,rowsMap_,fk_,partial_,partialK_,sums_,sumK_,fail_;
    size_t vertices_,tileCount_,donorCount_,regionCount_;
    std::vector<unsigned> regionFor_;
    std::string witness_;
    static void check(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
    template<class T> id<MTLBuffer> upload(const std::vector<T>& v) {
        check(!v.empty(),"empty material bed upload");
        auto b=[device_ newBufferWithBytes:v.data() length:v.size()*sizeof(T) options:MTLResourceStorageModeShared];
        check(b!=nil,"material bed allocation failed");return b;
    }
    id<MTLBuffer> allocate(size_t bytes) {
        auto b=[device_ newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        check(b!=nil,"material bed workspace allocation failed");return b;
    }
    id<MTLComputePipelineState> pipeline(id<MTLLibrary> lib,NSString* name) {
        auto f=[lib newFunctionWithName:name];check(f!=nil,"missing native registered bed kernel");
        NSError* e=nil;auto p=[device_ newComputePipelineStateWithFunction:f error:&e];
        check(p!=nil,"cannot create registered bed pipeline");return p;
    }
public:
    NumiHumanResolvedBed(id<MTLDevice> device,id<MTLLibrary> lib,const std::vector<mr_float4>& stock,
        std::span<const MRHumanRestingVertexMap> maps,std::span<const MRHumanRestingInfluence> influences,
        const std::vector<unsigned>& regions,unsigned count,unsigned stride,const char* witness):
        device_(device),vertices_(stock.size()),regionCount_(count),regionFor_(regions),witness_(witness) {
        check(regions.size()==stock.size() && maps.size()==stock.size(),"incompatible support layout");
        check(count==38 && (stride==1 || stride==32),"incompatible contact rows");
        std::map<Pair,std::vector<Term>> grouped;
        for(unsigned v=0;v<vertices_;++v) {
            if(stock[v].x==0.0f) continue;
            check(regions[v]<count && stock[v].x>0 && stock[v].y>0 && stock[v].z>0,"invalid bed column");
            double total=0;
            for(unsigned j=0;j<maps[v].influenceCount;++j) {
                const size_t ix=maps[v].firstInfluence+size_t(j)*stride;
                check(ix<influences.size(),"invalid material point binding");
                auto inf=influences[ix];auto w=inf.positionAndWeight;total+=w.w;
                if(w.w==0.0f) continue;
                check(w.w>0 && inf.body.x!=MR_INVALID_INDEX,"invalid material donor");
                grouped[{regions[v],inf.body.x}].push_back({{w.x*w.w,w.y*w.w,w.z*w.w,w.w},{v,0,0,0}});
            }
            check(std::abs(total-1)<1e-5,"material donor weights do not conserve traction");
        }
        std::vector<Term> t; std::vector<Pair> tile,row(count);
        std::vector<mr_uint4> donor;
        for(unsigned i=0;i<count;++i) {
            row[i][0]=unsigned(donor.size());
            for(auto it=grouped.lower_bound({i,0});it!=grouped.end() && it->first[0]==i;++it) {
                const unsigned start=unsigned(tile.size());
                const auto& group=it->second;
                for(size_t k=0;k<group.size();k+=128) {
                    const unsigned a=unsigned(t.size());const size_t end=std::min(k+128,group.size());
                    t.insert(t.end(),group.begin()+k,group.begin()+end);tile.push_back({a,unsigned(t.size())});
                }
                donor.push_back({it->first[1],start,unsigned(tile.size()),0});
            }
            row[i][1]=unsigned(donor.size());
            check(row[i][0]<row[i][1],"missing bed contact region");
        }
        columns_=pipeline(lib,@"nm_registered_bed_columns");tiles_=pipeline(lib,@"nm_registered_bed_tiles");
        donors_=pipeline(lib,@"nm_registered_bed_donors");rows_=pipeline(lib,@"nm_registered_bed_rows");
        check(tiles_.threadExecutionWidth==32,"bed reduction requires 32 lanes");
        tileCount_=tile.size();donorCount_=donor.size();
        stock_=upload(stock);terms_=upload(t);tilesMap_=upload(tile);donorsMap_=upload(donor);rowsMap_=upload(row);
        fk_=allocate(vertices_*2*sizeof(float));partial_=allocate(tile.size()*sizeof(mr_float4));
        partialK_=allocate(tile.size()*sizeof(float));sums_=allocate(donor.size()*sizeof(mr_float4));sumK_=allocate(donor.size()*sizeof(float));
        fail_=upload(std::vector<mr_float4>(vertices_));
    }
    ~NumiHumanResolvedBed() {
        const auto* failure=static_cast<const mr_float4*>(fail_.contents);
        unsigned first=UINT32_MAX,which=0,count=0;
        for(unsigned i=0;i<vertices_;++i) if(failure[i].x>0) {
            ++count;if(unsigned(failure[i].x)<first) {first=unsigned(failure[i].x);which=i;}
        }
        std::ofstream out(witness_);
        out<<std::setprecision(12)<<"{\"invalid_columns_observed\":"<<count;
        if(count)out<<",\"pre_dynamics_step\":"<<first-1u<<",\"vertex\":"<<which
            <<",\"region\":"<<regionFor_[which]<<",\"depth_m\":"<<failure[which].y
            <<",\"allowed_depth_m\":"<<failure[which].z<<",\"area_m2\":"<<failure[which].w;
        out<<"}\n";
    }
    void encode(id<MTLCommandBuffer> command,id<MTLBuffer> keys,id<MTLBuffer> contacts,id<MTLBuffer> points,
                id<MTLBuffer> low,id<MTLBuffer> jac,const MRHumanRestingSupportDispatchGPU& d,unsigned step) {
        auto begin=[&](id<MTLComputePipelineState> p,NSString* name){
            auto enc=[command computeCommandEncoder];check(enc!=nil,"cannot encode material contact");
            enc.label=name;[enc setComputePipelineState:p];return enc;
        };
        auto run=[&](id<MTLComputeCommandEncoder> enc,size_t n){
            [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[enc endEncoding];
        };
        auto e=begin(columns_,@"Bed material columns");
        [e setBuffer:stock_ offset:0 atIndex:0];[e setBuffer:keys offset:0 atIndex:1];
        [e setBuffer:fk_ offset:0 atIndex:2];[e setBuffer:fail_ offset:0 atIndex:3];
        [e setBytes:&step length:sizeof(step) atIndex:4];run(e,vertices_);
        e=begin(tiles_,@"Bed load quadrature");
        [e setBuffer:terms_ offset:0 atIndex:0];[e setBuffer:tilesMap_ offset:0 atIndex:1];
        [e setBuffer:fk_ offset:0 atIndex:2];[e setBuffer:partial_ offset:0 atIndex:3];
        [e setBuffer:partialK_ offset:0 atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(tileCount_,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];
        e=begin(donors_,@"Bed pressure moments per anatomical donor");
        [e setBuffer:donorsMap_ offset:0 atIndex:0];[e setBuffer:partial_ offset:0 atIndex:1];
        [e setBuffer:partialK_ offset:0 atIndex:2];[e setBuffer:sums_ offset:0 atIndex:3];
        [e setBuffer:sumK_ offset:0 atIndex:4];run(e,donorCount_);
        e=begin(rows_,@"Bed force and articulated virtual work");
        [e setBuffer:rowsMap_ offset:0 atIndex:0];[e setBuffer:donorsMap_ offset:0 atIndex:1];
        [e setBuffer:sums_ offset:0 atIndex:2];[e setBuffer:sumK_ offset:0 atIndex:3];
        [e setBuffer:contacts offset:0 atIndex:4];[e setBuffer:points offset:0 atIndex:5];
        [e setBuffer:low offset:0 atIndex:6];[e setBuffer:jac offset:0 atIndex:7];
        [e setBytes:&d length:sizeof(d) atIndex:8];
        [e dispatchThreads:MTLSizeMake(3u*(d.counts.z+1u),regionCount_,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];
    }
};
