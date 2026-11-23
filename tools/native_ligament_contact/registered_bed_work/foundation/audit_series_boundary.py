"""Check the displayed outer boundary against the separately reconstructed mattress face."""
import json
import subprocess
import sys
from pathlib import Path

from export_contact_columns import change
from material_context import sha


def build(base, dest, source):
    ids = json.loads((base / "identities.json").read_text())
    if dest.exists() or any(sha(base / name) != expected for name, expected in ids.items()):
        raise ValueError("Changed build input or existing destination")
    subprocess.run(["cp", "-cR", base, dest], check=True)
    p = dest / "human_respiration.metal"
    s = p.read_text()
    first = s.index("kernel void nm_human_resting_audit_skin_partials")
    last = s.index("kernel void nm_human_resting_reduce_skin_audit", first)
    kernel = s[first:last]
    kernel = change(kernel, "    uint lane [[thread_index_in_threadgroup]]", """    device const float4* stock [[buffer(6)]], constant uint4& materialRange [[buffer(7)]],
    device const MRHumanRestingInfluence* influences [[buffer(8)]],
    device const MRBodyStateGPU* bodies [[buffer(9)]],
    uint lane [[thread_index_in_threadgroup]]""")
    kernel = change(kernel, "            if(kUseRestingContouredBed) {", """            const bool registered = i >= materialRange.x && i - materialRange.x < materialRange.y;
            if(registered && stock[i-materialRange.x].x>0.f) {
                const float4 column=stock[i-materialRange.x];
                const auto mapping=map[i];
                float carrierZ=0.f;
                for(uint j=0;j<mapping.influenceCount;++j) {
                    const auto w=influences[mapping.firstInfluence+j*materialRange.z];
                    float height=w.positionAndWeight.z;
                    if(w.body.x!=MR_INVALID_INDEX) {
                        const auto body=bodies[w.body.x];
                        height=body.position.z+restingRotate(body.orientation,w.positionAndWeight.xyz).z;
                    }
                    carrierZ+=w.positionAndWeight.w*height;
                }
                const float depth=foamSurfaceZ-carrierZ;
                float sag=0.f;
                if(!isfinite(depth) || depth>bedClosure(column.yz,256u)) ++nonfinite;
                else if(depth>0.f) {
                    const auto interval=seriesBedInterval(column,depth);
                    sag=mix(seriesBed[interval.lo].z,seriesBed[interval.hi].z,interval.t);
                }
                z=vertices[i].position.z-(foamSurfaceZ-sag);
                // Positive pressure also forbids separation from the mattress.
                if(depth>0.f && z>.001f) ++count;
            } else if(kUseRestingContouredBed) {""")
    p.write_text(s[:first]+kernel+s[last:])
    p = dest / "NumiHumanRestingVisual.hpp"
    key = '        e.setBuffer(e.context,(__bridge void*)self.skinAuditPartials,0,3);'
    p.write_text(change(p.read_text(),key,key+"""
        const std::array<unsigned,4> materialRange={boundSkin[0],boundSkin[1],
            self.skinInfluenceLayout==NumiHumanSkinInfluenceLayout::tile32?32u:1u,0u};
        e.setBuffer(e.context,(__bridge void*)(self.foundationSkinStock ? self.foundationSkinStock : self.mapping),0,6);
        e.setBytes(e.context,materialRange.data(),sizeof(materialRange),7);
        e.setBuffer(e.context,(__bridge void*)self.influences,0,8);
        e.setBuffer(e.context,(__bridge void*)self.coupled.presentationBodies,0,9);"""))
    kernel = 'numiHumanRestingMakeSkinInfluenceFunction(lib,@"nm_human_resting_audit_skin_partials"'
    p.write_text(change(p.read_text(),kernel,kernel.replace('(lib,','(foundationSkinStock ? skinLibrary : lib,')))
    commands = [[v.replace(str(base)+"/",str(dest)+"/") for v in c] for c in json.loads((base/"build-commands.json").read_text())]
    cwd=Path(json.loads((source/"launch.json").read_text())["environment"]["DYLD_LIBRARY_PATH"]).parent
    for command in commands:
        subprocess.run(command,cwd=cwd,check=True)
    (dest/"build-commands.json").write_text(json.dumps(commands,indent=2)+"\n")
    (dest/"identities.json").write_text(json.dumps({k:sha(dest/k) for k in ids},indent=2)+"\n")


if __name__=="__main__":
    build(*(Path(s).resolve() for s in sys.argv[1:]))
