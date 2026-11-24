"""Build an optional, bounded observation of the live skin-to-body contact forces."""

import json
import subprocess
import sys
from pathlib import Path

from compile_connected_joint import digest


def one(s, old, new):
    if s.count(old) != 1:
        raise ValueError("Unexpected source owner: " + old[:90])
    return s.replace(old, new)


def build(case, host, gpu, out):
    cfg = json.loads((case / "launch.json").read_bytes())
    env = cfg["environment"]
    pins = cfg["pins"]
    if (
        Path(env["NUMI_MATTER_METALLIB_OVERRIDE"]).parent != gpu
        or Path(env["DYLD_LIBRARY_PATH"].split(":")[0]) != host
    ):
        raise ValueError("Different native ownership")
    original = host / "numi_human.mm"
    shader = gpu / "src/metal/numi_human.metalinc"
    program = host / "numi-human-native"
    if (
        pins[str(program)] != digest(program)
        or digest(host / "libmetalrobo.dylib") != pins[str(host / "libmetalrobo.dylib")]
        or digest(gpu / "NumiMatter.metallib") != pins[str(gpu / "NumiMatter.metallib")]
    ):
        raise ValueError("Unmatched native control")
    out.mkdir(exist_ok=True)
    dest = out / "src"
    (dest / "metal").mkdir(parents=True, exist_ok=True)
    for p in (gpu / "src").iterdir():
        if p.name == "metal":
            continue
        q = dest / p.name
        if p.name == "matter_kernels.metal":
            q.write_bytes(p.read_bytes())
        elif not q.exists():
            q.symlink_to(p.resolve())
    for p in shader.parent.iterdir():
        if p.name != shader.name and not (dest / "metal" / p.name).exists():
            (dest / "metal" / p.name).symlink_to(p.resolve())
    s = shader.read_text()
    k = "nm_numi_human_fem_body_contact_wrench_partial_cached"
    lo = s.index("kernel void " + k + "(")
    hi = s.index("\nkernel void ", lo + len(k))
    base = s[lo:hi]
    extra = base.replace(k, k + "_observed", 1)
    extra = one(
        extra,
        "    uint group [[threadgroup_position_in_grid]],",
        """    device atomic_uint* counts [[buffer(12)]],
    device float4* records [[buffer(13)]],
    uint group [[threadgroup_position_in_grid]],""",
    )
    match = "                value.torque=float4(cross(bodyPoint-pose.position.xyz,force),0.0f);"
    extra = one(
        extra,
        match,
        match
        + """
                if(environment==0u && dispatch.stepIndex<4u && closure>0.f) {
                    const uint slot=atomic_fetch_add_explicit(counts+dispatch.stepIndex,1u,memory_order_relaxed);
                    if(slot<8192u) {
                        device float4* q=records+7u*(8192u*dispatch.stepIndex+slot);
                        q[0]=as_type<float4>(uint4(row,sample.slaveNode,sample.bodyIndex,sample.flags));
                        q[1]=float4(slavePoint,closure);
                        q[2]=float4(bodyPoint,area);
                        q[3]=rowCache[environment*dispatch.femBodyContactSampleCount+row];
                        q[4]=value.force;
                        q[5]=value.torque;
                        q[6]=float4(pose.position.xyz,pressure);
                    }
                }""",
    )
    (dest / "metal" / shader.name).write_text(s[:hi] + "\n" + extra + s[hi:])
    s = original.read_text()
    member = "    __strong id<MTLBuffer> glutealObservation = nil;"
    s = one(
        s,
        member,
        member
        + """
    __strong id<MTLBuffer> routeCounts=nil,routeRecords=nil,routeWrenches=nil;
    std::string routePath;""",
    )
    selector = 'state_->femBodyContactWrenchPartialPipeline=pipeline("' + k + '");'
    s = one(
        s,
        selector,
        '''const char* route=std::getenv("NUMI_HUMAN_REGISTERED_FORCE_ROUTE_PREFIX");
                        if(route && *route) {
                            if(state_->environmentCount!=1u || pass.bodyPoseStride==0u) return false;
                            state_->routeCounts=[device newBufferWithLength:4u*sizeof(uint32_t) options:MTLResourceStorageModeShared];
                            state_->routeRecords=[device newBufferWithLength:4u*8192u*7u*sizeof(nm_float4) options:MTLResourceStorageModeShared];
                            state_->routeWrenches=[device newBufferWithLength:4u*pass.bodyPoseStride*sizeof(NMNumiHumanBodyWrenchGPU) options:MTLResourceStorageModeShared];
                            if(!state_->routeCounts || !state_->routeRecords || !state_->routeWrenches) return false;
                            std::memset(state_->routeCounts.contents,0,state_->routeCounts.length);
                            state_->routePath=route;
                        }
                        state_->femBodyContactWrenchPartialPipeline=pipeline(state_->routeCounts?"'''
        + k
        + '''_observed":"'''
        + k
        + """");""",
    )
    mark = "                        if(state_->contactRowCache)\n                            [encoder setBuffer:state_->femBodyContactRowCache offset:0u atIndex:11u];"
    s = one(
        s,
        mark,
        mark
        + """
                        if(state_->routeCounts) {
                            [encoder setBuffer:state_->routeCounts offset:0 atIndex:12u];
                            [encoder setBuffer:state_->routeRecords offset:0 atIndex:13u];
                        }""",
    )
    # A retried source step rewrites its own observation slot. Only published
    # acceptances can be paired to these diagnostic source generations.
    mark = "            const bool parallelWrench=state_->femBodyContactWrenchTileCount!=0u;"
    s = one(
        s,
        mark,
        mark
        + """
            if(state_->routeCounts && solverDispatch.stepIndex<4u) {
                id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];
                if(!blit) return false;
                [blit fillBuffer:state_->routeCounts range:NSMakeRange(solverDispatch.stepIndex*sizeof(uint32_t),sizeof(uint32_t)) value:0];
                [blit endEncoding];
            }""",
    )
    mark = "        if (state_->gluteal && !encodeKernel(state_->glutealPipeline"
    s = one(
        s,
        mark,
        """        if(state_->routeCounts && solverDispatch.stepIndex<4u) {
            const NSUInteger bytes=NSUInteger(pass.bodyPoseStride)*sizeof(NMNumiHumanBodyWrenchGPU);
            id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];
            if(!blit || state_->femBodyContactWrenchBuffer.length<bytes) return false;
            [blit copyFromBuffer:state_->femBodyContactWrenchBuffer sourceOffset:0 toBuffer:state_->routeWrenches destinationOffset:solverDispatch.stepIndex*bytes size:bytes];
            [blit endEncoding];
        }
"""
        + mark,
    )
    mark = "NumiHumanTendonFEMLoadAdapter::~NumiHumanTendonFEMLoadAdapter() {"
    s = one(
        s,
        mark,
        mark
        + """
    if(state_ && state_->routeCounts) {
        NSString* prefix=[NSString stringWithUTF8String:state_->routePath.c_str()];
        const uint32_t* counts=static_cast<const uint32_t*>(state_->routeCounts.contents);
        const auto* bytes=static_cast<const unsigned char*>(state_->routeRecords.contents);
        for(unsigned i=0;i<4u;++i) if(counts[i]) {
            NSString* name=[prefix stringByAppendingFormat:@"-%u.rows",i];
            const NSUInteger n=std::min(8192u,counts[i])*7u*sizeof(nm_float4);
            [[NSData dataWithBytes:bytes+i*8192u*7u*sizeof(nm_float4) length:n] writeToFile:name atomically:YES];
        }
        for(auto item:{std::pair<id<MTLBuffer>,NSString*>{state_->routeCounts,@"-counts.u32"},
                       std::pair<id<MTLBuffer>,NSString*>{state_->routeWrenches,@"-wrenches.f32"}}) {
            [[NSData dataWithBytes:item.first.contents length:item.first.length]
                writeToFile:[prefix stringByAppendingString:item.second] atomically:YES];
        }
    }""",
    )
    (out / original.name).write_text(s)
    cmds = [
        [v.replace(str(gpu) + "/", str(out) + "/") for v in c]
        for c in json.loads((gpu / "build.json").read_bytes())["commands"][:2]
    ]
    spec = json.loads((host / "build.json").read_bytes())
    c = spec["commands"][0]
    l = spec["direction"]["commands"][2]
    oldobj = c[c.index("-o") + 1]
    newobj = str(out / "numi_human.mm.o")
    c = [
        str(out / original.name) if v == str(original) else newobj if v == oldobj else v
        for v in c
    ]
    if str(out / original.name) not in c:
        raise ValueError("Unexpected host compile input")
    c[1:1] = ["-iquote", str(host)]
    l = [newobj if v == oldobj else v for v in l]
    l[l.index("-o") + 1] = str(out / "libmetalrobo.dylib")
    cmds += [c, l]
    for i, c in enumerate(cmds):
        with (out / f"compile-{i}.log").open("w") as log:
            subprocess.run(c, stdout=log, stderr=subprocess.STDOUT, check=True)
    r = dict(
        control=digest(case / "launch.json"),
        host=digest(original),
        gpu=digest(shader),
        commands=cmds,
        observed_host=digest(out / original.name),
        observed_gpu=digest(dest / "metal" / shader.name),
        library=digest(out / "libmetalrobo.dylib"),
        metal=digest(out / "NumiMatter.metallib"),
    )
    (out / "build.json").write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    build(*(Path(p).resolve() for p in sys.argv[1:]))
