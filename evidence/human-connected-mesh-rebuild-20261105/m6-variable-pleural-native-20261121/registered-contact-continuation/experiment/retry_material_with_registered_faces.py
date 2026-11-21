"""Retry a rejected material endpoint from its physical entry state with the newly observed faces."""
import shutil
from pathlib import Path
src=Path("feasible-nodal-owner");dst=Path("facet-retry-owner");dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file():shutil.copyfile(f,dst/f.name)
p=dst/"pelvic_end_geometry.metalinc";s=p.read_text().replace("bool test=!control[0]&&!needed[0]&&!needed[1];","bool test=!control[0] && (d.z || (!needed[0] && !needed[1]));")
s=s.replace("if(r[0]==0)control[0]=1;","if(r[0]==0)control[0]=(!needed[0] && !needed[1]);");p.write_text(s)
p=dst/"pelvic_contact.metalinc";p.write_text(p.read_text()+"""
// The coordinates and pressure multipliers at entry to the physical endpoint
// are one consistent seed. New face identities remain in the anatomical atlas.
kernel void nm_pelvic_retry_snapshot(constant uint4& d [[buffer(0)]],device const uint* needed [[buffer(1)]],
    device float4* coordinates [[buffer(2)]],device uint* slots [[buffer(3)]],device float* pressures [[buffer(4)]],
    device uint* included [[buffer(5)]],device float4* saved [[buffer(6)]],device uint2* records [[buffer(7)]],
    uint i [[thread_position_in_grid]]){
    if(!d.y){
        if(i<d.x)saved[i]=coordinates[i];
        if(i<512)records[i]=uint2(slots[i],as_type<uint>(pressures[i]));
        return;
    }
    if(d.y==1){
        if(i<512 && needed[i/256] && slots[i]!=UINT_MAX)included[slots[i]]=0;
        return;
    }
    if(i<d.x && needed[i>=pelvicFlex[1].x])coordinates[i]=saved[i];
    if(i<512 && needed[i/256]){
        uint2 s=records[i];slots[i]=s.x;pressures[i]=as_type<float>(s.y);
        if(s.x!=UINT_MAX)included[s.x]=1;
    }
}
""")
p=dst/"NumiHumanPelvicContact.hpp";s=p.read_text()
def before(needle,text):
    global s
    assert s.count(needle)==1,needle
    s=s.replace(needle,text+needle)
before('    std::map<unsigned,unsigned> spanIndex;',"""    id<MTLBuffer> retryCoordinates=nil,retryPorts=nil;
    id<MTLComputePipelineState> retrySnapshot=nil;
""")
before('            motionOwners=load(45);',"""            retryCoordinates=alloc(countNodes*16);retryPorts=alloc(512*8);
            retrySnapshot=pipeline(@"nm_pelvic_retry_snapshot");
""")
key='        if(!post){\n            const bool seed'
before(key,"""        auto snapshotEndpoint=[&](unsigned action){
            [e setComputePipelineState:retrySnapshot];words(0,{countNodes,action,0,0});
            bind(1,needed);bind(2,coordinates);bind(3,poolSlots);bind(4,pressures);
            bind(5,included);bind(6,retryCoordinates);bind(7,retryPorts);
            run(action==1?4u:(countNodes+127)/128,128);
        };
        if constexpr(pelvicMotionRegistry){if(post)snapshotEndpoint(0);}
""")
key="words(0,{nCrossingGroups,unsigned(outer==UINT_MAX),0,0})"
assert s.count(key)==1
s=s.replace(key,"words(0,{nCrossingGroups,unsigned(outer==UINT_MAX),unsigned(outer>=51 && outer!=UINT_MAX),0})")
a=s.index("        for(unsigned outer=0;outer<16;++outer){")
b=s.index("        [e setComputePipelineState:gate];",a)
loops=s[a:b]
assert loops.count("certify(outer);")==1
loops=loops.replace("certify(outer);","certify(retry?64+outer:outer);").replace("certify(16u+2*sweep)","certify((retry?80u:16u)+2*sweep)").replace("certify(17u+2*sweep)","certify((retry?81u:17u)+2*sweep)").replace("certify(48+local)","certify((retry?112u:48u)+local)")
s=s[:a]+"        auto solveEndpoint=[&](bool retry){\n"+loops+"""        };
        solveEndpoint(false);
        if constexpr(pelvicMotionRegistry){
            if(post){
                unsigned count=unsigned(spanIndex.size());
                [e setComputePipelineState:makeSpans];[e setBytes:&count length:4 atIndex:0];
                bind(1,needed);bind(2,dispatchSizes);bind(3,dispatchArgs);run((count+31)/32,32);
                optional=true;
                snapshotEndpoint(1);snapshotEndpoint(2);
                observe(1,1);bulkForces();select(1,false);nearest(0,1);compliance();
                [e setComputePipelineState:candidatePressure];words(0,{1,0,0,0});
                bind(1,poolSlots);bind(2,rows);bind(3,gram);bind(4,pressures);bind(5,stats);bind(6,needed);run(2,32);
                certify(63);optional=false;
                solveEndpoint(true);
            }
        }
"""+s[b:]
p.write_text(s)
