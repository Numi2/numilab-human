"""Recover fixed-geometry nodal equilibrium with a unilateral force fit after a failed contact solve."""
import json
from pathlib import Path
import subprocess
import sys
from build_distinct_registry import replace
from emit_skin_source_registration import sha


def main(src,app,out):
    pins=json.loads((src/'identities.json').read_text())
    if any(sha(Path(f))!=v for f,v in pins.items()):raise ValueError('unmatched shared-registry build')
    out.mkdir(exist_ok=False);(out/'metalrobo').mkdir()
    (out/'metalrobo/MetalRobo.metallib').hardlink_to(src/'metalrobo/MetalRobo.metallib')
    for f in src.iterdir():
        if f.is_file() and f.suffix in ('.mm','.hpp','.inc','.metal','.metalinc'):
            (out/f.name).hardlink_to(f)
    kernel=out/'pelvic_contact.metalinc';code=kernel.read_text()
    code=replace(code,'device float4* residual [[buffer(9)]], uint side','device float4* residual [[buffer(9)]], constant uint& unilateral [[buffer(10)]], uint side')
    old='''        float d = 0;
        if (id != UINT_MAX && isfinite(n2) && n2 > .5f &&
            (pressure[i] > 0 || rows[i].leverB.w <= 0)) {'''
    new='''        float d = 0;
        float gap=id==UINT_MAX?INFINITY:rows[i].leverB.w;
        // At a fixed pose a separated, compressive-only contact cannot support a load.
        // The regular nonlinear path can still change coordinates to close this gap.
        if(unilateral && id!=UINT_MAX && isfinite(gap) && gap>=5.e-7f)pressure[i]=0.f;
        if (id != UINT_MAX && isfinite(n2) && n2 > .5f &&
            (pressure[i] > 0 || gap <= (unilateral?2.e-9f:0.f))) {'''
    code=replace(code,old,new)
    code=replace(code,'    for (uint j = lane; j < part.y; j += 32) {\n        uint node = part.x + j;\n        float3 force = internalForce[node].xyz;',
                 '    if(unilateral)threadgroup_barrier(mem_flags::mem_device);\n    for (uint j = lane; j < part.y; j += 32) {\n        uint node = part.x + j;\n        float3 force = internalForce[node].xyz;')
    code+='''
kernel void nm_pelvic_unilateral_recovery_span(device const float4* stats [[buffer(0)]],
    device const float4* quality [[buffer(1)]],device uint* recover [[buffer(2)]],
    uint side [[thread_position_in_grid]]){
    if(side<2){
        float4 q=quality[side];
        // Invalid geometry and compressive penetrations go directly to the regular gate.
        recover[side]=all(isfinite(q)) && all(isfinite(stats[side])) && stats[side].x<=2.e-9f &&
            q.z>=5.e-7f && !pelvicEquilibrated(stats[side],q);
    }
}
'''
    kernel.unlink();kernel.write_text(code)
    header=out/'NumiHumanPelvicContact.hpp';s=header.read_text()
    s=replace(s,'motionPhase=nil,motionRegister=nil,motionBind=nil;','motionPhase=nil,motionRegister=nil,motionBind=nil,unilateralSpan=nil;')
    s=replace(s,'motionPhase=pipeline(@"nm_pelvic_motion_phase");','unilateralSpan=pipeline(@"nm_pelvic_unilateral_recovery_span");motionPhase=pipeline(@"nm_pelvic_motion_phase");')
    s=replace(s,'bind(9,tractionResidual);run(2,32);','bind(9,tractionResidual);words(10,{0,0,0,0});run(2,32);')
    old='''                solveEndpoint(true);
            }'''
    new='''                solveEndpoint(true);
                // Use the existing contact and residual owners at the final geometry.
                // Successful endpoints skip the fit and all its residual measurements.
                [e setComputePipelineState:unilateralSpan];bind(0,stats);bind(1,quality);
                bind(2,materialNeeded);run(1,32);
                [e setComputePipelineState:makeSpans];[e setBytes:&count length:4 atIndex:0];
                bind(1,materialNeeded);bind(2,dispatchSizes);bind(3,dispatchArgs);run((count+31)/32,32);
                optional=true;
                [e setComputePipelineState:fitTractions];bind(0,materialNeeded);bind(1,poolSlots);bind(2,pool);
                bind(3,localN);bind(4,rows);bind(5,kr);bind(6,bulkGrad);bind(7,pressures);bind(8,inc);
                bind(9,tractionResidual);words(10,{1,0,0,0});run(2,32);
                [e setComputePipelineState:candidatePressure];words(0,{1,0,0,0});bind(1,poolSlots);
                bind(2,rows);bind(3,gram);bind(4,pressures);bind(5,stats);bind(6,materialNeeded);run(2,32);
                // Rebuild incidence even for newly loaded touching sites and certify the fresh forces.
                // Surface coordinates have not changed; the mandatory full-pair endpoint test follows.
                [e setComputePipelineState:clearInc];[e setBytes:&countNodes length:4 atIndex:0];bind(1,inc);run((countNodes*8+127)/128,128);
                [e setComputePipelineState:buildInc];bind(0,poolSlots);bind(1,pressures);bind(2,pool);bind(3,inc);run(16,128);
                [e setComputePipelineState:measureForce];words(0,{0,residualGroups,1,0});bind(1,kr);bind(2,kc);bind(3,kv);bind(4,coordinates);
                bind(5,inc);bind(6,poolSlots);bind(7,pressures);bind(8,localN);bind(9,pool);bind(10,partial);bind(11,materialNeeded);
                bind(12,bulkGrad);bind(13,attachment);run(2*residualGroups,128);
                [e setComputePipelineState:confirm];bind(0,samples);bind(1,rows);bind(2,gram);bind(3,coordinates);bind(4,pressures);
                bind(5,stats);words(6,{p.stepIndex,countTrace,116,1});bind(7,trace);bind(8,materialNeeded);bind(9,history);
                bind(10,poolMin);bind(11,poolRegions);[e setBytes:&numRegions length:4 atIndex:12];bind(13,partial);
                [e setBytes:&residualGroups length:4 atIndex:14];bind(15,quality);bind(16,qualityTrace);bind(17,iterationTrace);run(2,32);
                optional=false;
            }'''
    s=replace(s,old,new);header.unlink();header.write_text(s)
    cmds=[[z.replace(str(src)+'/',str(out)+'/') for z in c] for c in json.loads((src/'build-commands.json').read_text())]
    (out/'build-commands.json').write_text(json.dumps(cmds,indent=2)+'\n')
    for c in cmds:subprocess.run(c,cwd=app/'build-native',check=True)
    (out/'identities.json').write_text(json.dumps({str(f):sha(f) for f in (kernel,header,out/'numi-human-native',out/'metalrobo/HumanRestingSkin.metallib')},indent=2)+'\n')

if __name__=='__main__':main(*(Path(x).resolve() for x in sys.argv[1:]))
