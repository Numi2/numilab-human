// Qualification harness only: the linked Matter runtime owns every physical step.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "numi/matter/human_physiology.hpp"
#include "metalrobo/engine_types.h"
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace numi::matter;
namespace {
void need(bool ok, const std::string& message) { if(!ok) throw std::runtime_error(message); }
NSDictionary* json(const char* path) {
    NSData* bytes=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
    id value=bytes ? [NSJSONSerialization JSONObjectWithData:bytes options:0 error:nil] : nil;
    need([value isKindOfClass:NSDictionary.class], "cannot read payload object");
    return value;
}
void associationOnly(const char* original,const char* associated) {
    // Native loading below separately rejects duplicate JSON fields and types.
    NSDictionary* before=json(original); NSDictionary* after=json(associated);
    NSMutableDictionary* restored=[after mutableCopy];
    for(NSString* key in @[@"model_id",@"authored_graph_sha256",@"source_graph_sha256"]) {
        need(![before[key] isEqual:after[key]], "association identity failed to change");
        restored[key]=before[key];
    }
    NSArray* oldRows=before[@"compartments"];
    NSMutableArray* rows=[after[@"compartments"] mutableCopy];
    need(oldRows.count==21 && rows.count==21,"compartment count changed");
    const unsigned indices[]={15,16,19,20};
    NSArray* labels=@[@"right_atrium",@"right_ventricle",@"left_atrium",@"left_ventricle"];
    NSArray* fma=@[@"FMA:11359",@"FMA:9291",@"FMA:9465",@"FMA:9466"];
    for(unsigned i=0;i<4;++i) {
        unsigned index=indices[i]; NSMutableDictionary* row=[rows[index] mutableCopy];
        NSString* source=[@"source_aggregate:CVSim21:" stringByAppendingString:labels[i]];
        need([oldRows[index][@"anatomical_region_id"] isEqual:source],"baseline source chamber differs");
        need([row[@"anatomical_region_id"] isEqual:fma[i]],"cavity association differs");
        row[@"anatomical_region_id"]=oldRows[index][@"anatomical_region_id"]; rows[index]=row;
    }
    restored[@"compartments"]=rows;
    need([restored isEqual:before],"anatomy association changed physical parameters or unrelated fields");
}
struct Run {
    CompiledWorld world; Runtime runtime; id<MTLDevice> device; id<MTLCommandQueue> queue; id<MTLBuffer> statuses;
    explicit Run(const char* input) {
        WorldSource source; source.environmentCount=2; source.frameTimestep=.001; source.gravity={0,0,0};
        source.mixedSolver.relativeResidual=1e-7; source.mixedSolver.newtonIterations=12; source.mixedSolver.fgmresIterations=64;
        std::string error; need(readHumanPhysiologyNetwork(input,source.vascular,&error),error);
        auto compiled=compileWorld(source,{.maximumRateExponent=0});
        for(const auto& d:compiled.diagnostics) if(d.severity==Diagnostic::Severity::error) error+=d.message+"; ";
        need(compiled.succeeded(),error);
        const auto path=std::filesystem::temp_directory_path()/(std::string("cavity-")+[[NSUUID UUID] UUIDString].UTF8String+".nmatterpack");
        need(writePackage(compiled,path,&error),error);
        const bool loaded=readPackage(path,world,nullptr,&error);
        std::error_code removal;std::filesystem::remove(path,removal);
        need(loaded&&!removal&&world.fingerprint==compiled.world.fingerprint,"package reload failed: "+error);
        device=MTLCreateSystemDefaultDevice(); need(device!=nil,"Metal unavailable");
        need([[device name] rangeOfString:@"Apple"].location!=NSNotFound && [[device name] rangeOfString:@"Paravirtual"].location==NSNotFound,"physical Apple Metal required");
        queue=[device newCommandQueue]; need(queue!=nil,"queue unavailable");
        statuses=[device newBufferWithLength:2*sizeof(MRMetalWorldStatusGPU) options:MTLResourceStorageModeShared];
        need(statuses!=nil,"status buffer unavailable");
        RuntimeConfiguration cfg;cfg.metallib=NUMI_MATTER_METALLIB;cfg.environmentCount=2;cfg.captureEvents=false;cfg.captureDiagnostics=true;cfg.adaptiveTransfer=false;
        auto initialized=runtime.initialize(world,cfg);need(initialized.encoded,initialized.message);
        std::cout<<"device="<<[device name].UTF8String<<" fingerprint="<<world.fingerprint<<'\n';
    }
    RuntimeStateSnapshot snapshot(){auto s=runtime.snapshot();need(s.available,s.message);return s;}
    void step(unsigned index) {
        auto* out=static_cast<MRMetalWorldStatusGPU*>(statuses.contents);
        for(unsigned e=0;e<2;++e){out[e]={};out[e].environment=e;}
        id<MTLCommandBuffer> cb=[queue commandBuffer];need(cb!=nil,"command buffer unavailable");
        EncodeRequest req;req.commandBuffer=(__bridge void*)cb;req.environmentStatuses=(__bridge void*)statuses;
        req.controlStep=index;req.physicsSubsteps=1;req.timestepSeconds=runtime.timestepSeconds();req.runAdaptiveTransfer=false;
        req.phase=EncodePhase::preDynamics;auto encoded=runtime.encode(req);need(encoded.encoded,encoded.message);
        req.phase=EncodePhase::postCommit;encoded=runtime.encode(req);need(encoded.encoded,encoded.message);
        [cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"Metal command failed");
        const auto s=snapshot();need(s.statuses.size()==2,"status count differs");
        for(const auto& status:s.statuses) need(status.code==NM_STATUS_SUCCESS,"physical step rejected");
    }
};
void equal(const RuntimeStateSnapshot& a,const RuntimeStateSnapshot& b) {
    need(a.vascularState.size()==90&&b.vascularState.size()==90,"vascular state size differs");
    need(a.vascularClock.size()==2&&b.vascularClock.size()==2,"vascular clock size differs");
    for(const auto* snapshot:{&a,&b}) for(const auto& value:snapshot->vascularState)
        need(std::isfinite(value.x)&&std::isfinite(value.y)&&std::isfinite(value.z)&&std::isfinite(value.w),"nonfinite vascular state");
    need(std::memcmp(a.vascularState.data(),b.vascularState.data(),90*sizeof(nm_float4))==0,"anatomical association changed physical state");
    need(std::memcmp(a.vascularClock.data(),b.vascularClock.data(),2*sizeof(NMVascularClockGPU))==0,"accepted clocks differ");
    need(std::memcmp(a.vascularState.data(),a.vascularState.data()+45,45*sizeof(nm_float4))==0,"baseline environment pair differs");
    need(std::memcmp(b.vascularState.data(),b.vascularState.data()+45,45*sizeof(nm_float4))==0,"registered environment pair differs");
}
}
int main(int argc,const char* argv[]) {@autoreleasepool{try {
    need(argc==3,"usage: cardiac-cavity-native-check BASELINE.json ASSOCIATED.json");
    // Both parsers complete before the NSDictionary equality comparison.
    Run baseline(argv[1]),associated(argv[2]);associationOnly(argv[1],argv[2]);
    need(baseline.world.fingerprint!=associated.world.fingerprint,"anatomical identity absent from package fingerprint");
    auto a=baseline.snapshot(),b=associated.snapshot();equal(a,b);
    const auto initial=a;
    const int exponent=std::bit_cast<std::int32_t>(baseline.world.vascular.layout.clock.x);
    const auto ticks=static_cast<std::uint64_t>(std::ldexp(double(baseline.runtime.timestepSeconds()),-exponent));
    need(ticks>0,"accepted clock has no timestep ticks");
    for(unsigned i=0;i<64;++i){@autoreleasepool{
        baseline.step(i);associated.step(i);auto nextA=baseline.snapshot(),nextB=associated.snapshot();equal(nextA,nextB);
        const unsigned __int128 expected=static_cast<unsigned __int128>(ticks)*(i+1u);
        for(const auto& clock:nextB.vascularClock)
            need(clock.low==std::uint64_t(expected)&&clock.high==std::uint64_t(expected>>64u),"accepted time did not advance exactly");
        if(i==32) {
            need(associated.runtime.restore(b).encoded,"snapshot restore rejected");associated.step(i);equal(nextA,associated.snapshot());
            need(!associated.runtime.restore(a).encoded,"cross-anatomy snapshot identity was admitted");equal(nextA,associated.snapshot());
        }
        a=std::move(nextA);b=std::move(nextB);
    }}
    bool evolved=false;
    for(unsigned i=0;i<45;++i) evolved=evolved || a.vascularState[i].x!=initial.vascularState[i].x;
    need(evolved,"vascular state did not evolve during accepted steps");
    std::cout<<"cavity_native_check=pass accepted_steps=64 environments_per_model=2 failed_steps=0 dt_seconds="<<associated.runtime.timestepSeconds()
             <<" physical_parameters=identical state_pair=bitwise environment_pair=bitwise accepted_clocks=bitwise snapshot_replay=bitwise cross_identity_restore=rejected package_fingerprints=distinct finite_state=true state_evolved=true mechanical_mass_added=0 anatomical_disjointness=unqualified physiological_calibration=unqualified\n";
    return 0;
} catch(const std::exception& error) {std::cerr<<"cavity_native_check=failed reason="<<error.what()<<'\n';return 1;}}}
