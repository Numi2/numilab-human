"""Use the already allocated bilateral facet pool for either anatomical owner."""
import json
from pathlib import Path
import subprocess
import sys
from build_distinct_registry import replace
from emit_skin_source_registration import sha


def main(src,app,out):
    pins=json.loads((src/'identities.json').read_text())
    for path,wanted in pins.items():
        if sha(Path(path))!=wanted:raise ValueError('unmatched registry source '+path)
    out.mkdir(exist_ok=False)
    (out/'metalrobo').mkdir()
    for f in src.iterdir():
        if f.is_file() and f.suffix in ('.metalinc','.metal','.mm','.hpp','.inc'):
            (out/f.name).hardlink_to(f)
    (out/'metalrobo/MetalRobo.metallib').hardlink_to(src/'metalrobo/MetalRobo.metallib')
    def save(name,s):
        f=out/name;f.unlink();f.write_text(s)
    registry='pelvic_motion_registry.metalinc';s=(src/registry).read_text()
    a=s.index('kernel void nm_pelvic_motion_register(');b=s.index('kernel void nm_pelvic_motion_bind(',a)
    s=s[:a]+Path(__file__).with_name('pelvic_shared_slot_register.metalinc').read_text()+'\n'+s[b:]
    s=replace(s,'for(uint j=0;j<256;++j){\n        uint slot=side*256+j;','for(uint slot=0;slot<512;++slot){')
    s=replace(s,'uint i=atomic_fetch_add_explicit(metrics+side,1u,memory_order_relaxed);\n    if(i<256)queue[256*side+i]=key;',
              'atomic_fetch_add_explicit(metrics+side,1u,memory_order_relaxed);\n    uint i=atomic_fetch_add_explicit(metrics+5,1u,memory_order_relaxed);\n    if(i<512)queue[i]=key;')
    s=replace(s,'if(i<5)atomic_store_explicit','if(i<6)atomic_store_explicit')
    s=replace(s,'}else if(i<2)atomic_store_explicit(metrics+i,0u','}else if(i<2||i==5)atomic_store_explicit(metrics+i,0u')
    s=replace(s,'device PelvicPoolSample* pool [[buffer(6)]],','device PelvicPoolSample* pool [[buffer(6)]],device uint4* regions [[buffer(7)]],constant uint& firstRegion [[buffer(8)]],')
    s=replace(s,'if(k>=512)return;uint2 key=keys[k];if(!key.x)return;',
              'if(k>=512)return;uint2 key=keys[k];if(!key.x){regions[firstRegion+k].z=256u*(k/256u);return;}')
    s=replace(s,'uint4 bone=owners[b];auto f=','uint4 bone=owners[b];regions[firstRegion+k].z=256u*bone.z;auto f=')
    save(registry,s)
    name='NumiHumanPelvicContact.hpp';s=(src/name).read_text()
    s=replace(s,'motionMetrics=alloc(20,true);','motionMetrics=alloc(24,true);')
    s=replace(s,'bind(6,pool);run(4,128);\n        };','bind(6,pool);bind(7,poolRegions);unsigned firstDynamic=numRegions-512;[e setBytes:&firstDynamic length:4 atIndex:8];run(4,128);\n        };')
    s=replace(s,'bind(3,liveExtras);run(2,256);bindMotion();','bind(3,liveExtras);bind(4,motionOwners);run(1,512);bindMotion();')
    s=replace(s,'if constexpr(pelvicMotionRegistry){\n            if(std::getenv','''if constexpr(pelvicMotionRegistry){
            motionOwners=load(45);
            if(numRegions<512||finiteCount<512)throw std::runtime_error("undersized dynamic pelvic region table");
            const uint32_t* table=static_cast<const uint32_t*>(poolRegions.contents);
            for(unsigned k=0;k<512;++k){
                auto region=table+4*(numRegions-512+k);
                if(region[0]!=poolCount-(512-k)*16 || region[1]!=16 || region[2]!=256*(k/256))
                    throw std::runtime_error("unexpected dynamic contact pool ownership");
            }
            if(std::getenv''')
    s=replace(s,'motionOwners=load(45);registeredFaces=load(46);','registeredFaces=load(46);')
    s=replace(s,'const auto* regions=static_cast<const uint32_t*>(poolRegions.contents);','''auto* regions=static_cast<uint32_t*>(poolRegions.contents);
                const auto* cortical=static_cast<const uint32_t*>(motionOwners.contents);
                for(unsigned k=0;k<512;++k){
                    uint32_t b=keys[2*k],m=keys[2*k+1];
                    if((b==0)!=(m==0)||b>motionOwners.length/16||m>numFaces)
                        throw std::runtime_error("initial feature outside its anatomical surface");
                    if(b)regions[4*(numRegions-512+k)+2]=256u*cortical[4*(b-1)+2];
                }''')
    save(name,s)
    cmds=[[a.replace(str(src)+'/',str(out)+'/') for a in c] for c in json.loads((src/'build-commands.json').read_text())]
    (out/'build-commands.json').write_text(json.dumps(cmds,indent=2)+'\n')
    for c in cmds:subprocess.run(c,cwd=app/'build-native',check=True)
    (out/'identities.json').write_text(json.dumps({str(out/k):sha(out/k) for k in
        (name,registry,'pelvic_end_geometry.metalinc','metalrobo/HumanRestingSkin.metallib','numi-human-native')},indent=2)+'\n')

if __name__=='__main__':main(*(Path(a).resolve() for a in sys.argv[1:]))
