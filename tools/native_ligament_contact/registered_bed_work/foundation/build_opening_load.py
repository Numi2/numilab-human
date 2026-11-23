"""Capture the applied bed generalized force once from the native material operator."""
import json,shutil,subprocess,sys
from pathlib import Path
from material_context import sha
from export_contact_columns import change
def build(base,out,source):
    ids=json.loads((base/'identities.json').read_text())
    if any(sha(base/k)!=v for k,v in ids.items()):raise ValueError('changed owner')
    subprocess.run(['cp','-cR',base,out],check=True)
    def edit(name,x,y):
        p=out/name;p.write_text(change(p.read_text(),x,y))
    n='NumiHumanResolvedBed.metalinc'
    edit(n,'    uint2 id [[thread_position_in_grid]]) {','''    device float* openingForce [[buffer(9)]], constant uint& step [[buffer(10)]],
    uint2 id [[thread_position_in_grid]]) {''')
    edit(n,'    if (scalar==0u) {','''    if(step==0u && axis==2u)
        openingForce[row*(nv+1u)+component] = component==nv ? F : action;
    if (scalar==0u) {''')
    n='NumiHumanResolvedBed.hpp'
    edit(n,'    size_t vertices_,','    id<MTLBuffer> openingForce_;\n    size_t openingStride_=0;\n    size_t vertices_,')
    edit(n,'        auto begin=[&]', '''        if(!openingForce_) {
            openingStride_=size_t(d.counts.z)+1u;
            openingForce_=upload(std::vector<float>(regionCount_*openingStride_,NAN));
        }
        auto begin=[&]''')
    edit(n,'        [e dispatchThreads:MTLSizeMake(3u*','''        [e setBuffer:openingForce_ offset:0 atIndex:9];
        [e setBytes:&step length:sizeof(step) atIndex:10];
        [e dispatchThreads:MTLSizeMake(3u*''')
    edit(n,'        const auto* failure=','''        if(openingForce_) {
            const auto values=static_cast<const float*>(openingForce_.contents);
            std::ofstream table(witness_+".opening.csv");
            table<<std::setprecision(12)<<"row";
            for(size_t j=0;j+1u<openingStride_;++j)table<<",q"<<j;
            table<<",normal_N\\n";
            for(size_t r=0;r<regionCount_;++r) {
                table<<r;
                for(size_t j=0;j<openingStride_;++j)table<<','<<values[r*openingStride_+j];
                table<<'\\n';
            }
        }
        const auto* failure=''')
    cmds=[[v.replace(str(base)+'/',str(out)+'/') for v in c] for c in json.loads((base/'build-commands.json').read_text())]
    cwd=Path(json.loads((source/'launch.json').read_text())['environment']['DYLD_LIBRARY_PATH']).parent
    for cmd in cmds:subprocess.run(cmd,cwd=cwd,check=True)
    (out/'build-commands.json').write_text(json.dumps(cmds,indent=2)+'\n')
    (out/'identities.json').write_text(json.dumps({k:sha(out/k) for k in ids},indent=2)+'\n')
if __name__=='__main__':build(*(Path(s).resolve() for s in sys.argv[1:]))
