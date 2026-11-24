"""Read the bounded GPU direction outcomes in the existing selected backtracking snapshot."""
import json,os,re,shutil,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from build_coupled_volume_direction import replace_once
from compile_connected_joint import digest

def build(out):
    parent=out.parent;base=parent/'incident-direction';app=parent/'bound-incident-app'
    out.mkdir();rt=out/'runtime';rt.mkdir();(out/'app').mkdir();src=base/'metal/src';dst=out/'metal/src';(dst/'metal').mkdir(parents=True)
    for f in src.iterdir():
        if f.name=='metal':continue
        if f.name=='matter_kernels.metal':shutil.copy2(f,dst/f.name)
        else:(dst/f.name).symlink_to(f.resolve(),target_is_directory=f.is_dir())
    for f in (src/'metal').iterdir():
        if f.name!='fem.metalinc':(dst/'metal'/f.name).symlink_to(f.resolve())
    os.link(base/'runtime/libmetalrobo.dylib',rt/'libmetalrobo.dylib')
    s=(src/'metal/fem.metalinc').read_text()
    i=s.index('inline void registeredVolumeCellDirection(');e=s.index('kernel void nm_fem_registered_volume_active_incident(',i)
    helper=s[i:e].replace('inline void ','inline uint ',1).replace('return;','return 1u;')
    labels=[('if(privateCount<2u)return 1u;',2),('if(p[k].restAndFixed.w>.5f || !(p[k].positionAndMass.w>0.f)) return 1u;',3),('if (!(current>=lo && current<=hi)) return 1u;',4),('if(J>lo && J<hi)return 1u;',5),('return 1u;\n        }\n        const float err=target-J;',6),('if(!(abs(err)<lastError))return 1u;',7),('if(!(actual>lo && actual<hi))return 1u;',8),('if(!(denominator>0.f) || !isfinite(denominator))return 1u;',9)]
    for text,reason in labels:helper=replace_once(helper,text,text.replace('return 1u;',f'return {reason}u;'))
    helper=helper.rstrip();assert helper.endswith('}');helper=helper[:-1]+'    return 10u;\n}\n\n'
    s=s[:i]+helper+s[e:];old=re.search(r'if\(color\[i\]==wave\)\{ (registeredVolumeCellDirection\([^;]+\)); \}',s).group(0)
    call=old.split('{ ',1)[1].split(';',1)[0]
    s=replace_once(s,old,'''if(color[i]==wave){
                const uint reason=CALL;
                // Only the sparse diagnostic suffix is written; the physical row list is immutable.
                ((device uint*)rows)[count+1u+i]=sorted[i]|(reason<<24u);
            }'''.replace('CALL',call))
    (dst/'metal/fem.metalinc').write_text(s)
    a=(app/'app-runtime.mm').read_text();a=replace_once(a,'partialBytes+sizeof(nm_float4) options:MTLResourceStorageModeShared];','partialBytes+sizeof(nm_float4)+33u*sizeof(uint32_t) options:MTLResourceStorageModeShared];')
    a=replace_once(a,'toBuffer:snapshot destinationOffset:partialBytes size:sizeof(nm_float4)];','''toBuffer:snapshot destinationOffset:partialBytes size:sizeof(nm_float4)];
                    const bool traceVolume=state.volumeDirection && state.volumeHints && objectIndex==state.volumeOwner;
                    if(traceVolume)[copy copyFromBuffer:state.volumeHints sourceOffset:state.volumeTileCount*sizeof(uint32_t)
                        toBuffer:snapshot destinationOffset:partialBytes+sizeof(nm_float4) size:33u*sizeof(uint32_t)];''')
    a=replace_once(a,'const nm_float4 selected=records[partialBytes/sizeof(nm_float4)];','''const nm_float4 selected=records[partialBytes/sizeof(nm_float4)];
                        if(traceVolume){
                            const uint32_t* sparse=(const uint32_t*)((const char*)snapshot.contents+partialBytes+sizeof(nm_float4));
                            std::ostringstream message;
                            message<<"registered_volume_direction_outcomes root="<<root<<" iteration="<<iteration<<" live="<<sparse[0]<<" tile_reason=";
                            if(sparse[0]<=32u)for(uint32_t i=0;i<sparse[0];++i)message<<(i?",":"")<<(sparse[i+1]&0xffffffu)<<"/"<<(sparse[i+1]>>24u);
                            std::fprintf(stderr,"%s\\n",message.str().c_str());
                        }''')
    ap=out/'app';(ap/'app-runtime.mm').write_text(a)
    prior=json.loads((base/'build.json').read_bytes());link=json.loads((app/'build.json').read_bytes())
    gpu=[[v.replace(str(base/'metal')+'/',str(out/'metal')+'/')for v in c]for c in prior['commands'][-2:]]
    cpu=[[v.replace(str(app)+'/',str(ap)+'/').replace(str(base/'runtime')+'/',str(rt)+'/')for v in c]for c in link['commands']]
    def execute(item):
        name,commands=item
        with (out/(name+'.log')).open('w')as f:
            for c in commands:subprocess.run(c,stdout=f,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(2) as pool:list(pool.map(execute,[('metal',gpu),('app-build',cpu)]))
    for key,f in [('shader',out/'metal/NumiMatter.metallib'),('shader_source',dst/'metal/fem.metalinc')]:prior[key]=digest(f)
    prior['commands']=cpu+gpu;(out/'build.json').write_text(json.dumps(prior,indent=2)+'\n')
    link.update(commands=cpu,app=digest(ap/'numi-human-native'),source=digest(ap/'app-runtime.mm'));(ap/'build.json').write_text(json.dumps(link,indent=2)+'\n')
if __name__=='__main__':build(Path(sys.argv[1]).resolve())
