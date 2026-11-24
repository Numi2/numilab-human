"""Identify the compiled tissue by its complete ordered connectivity and material assignments."""
import sys,json,hashlib,subprocess,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from compile_connected_joint import digest
from make_shared_tissue_contact import substitute

def built(job,original,base,material,out):
    rd=lambda p:json.loads(p.read_bytes());before=rd(base/'build.json');src=rd(material/'result.json')
    elements=material/'mesh.ele';nodes=material/'mesh.node'
    if digest(elements)!=src['unchanged_topology'] or digest(nodes)!=src['material'] or digest(material/'current.node')!=src['initial']:raise ValueError('Unregistered material')
    table=np.loadtxt(elements,dtype='<u4',skiprows=1);num=int(nodes.open().readline().split()[0])
    if table.shape[1]!=6 or np.any(table[:,0]!=np.arange(1,len(table)+1)) or np.min(table[:,1:5])<1 or np.max(table[:,1:5])>num or set(np.unique(table[:,5]))!={15,16}:raise ValueError('Unknown source cell schema')
    local=table[:,1:].copy();local[:,:4]-=1
    promised=hashlib.sha256(local.tobytes()).hexdigest()
    check='''const bool certifiedMaterial=geometry && std::string_view(geometry)=="ELEMENTS";
                if(certifiedMaterial){
                    // Authenticate every local node index and material assignment against
                    // the validated input, after the compiler has assigned rigid/FEM offsets.
                    for(uint32_t owner=0;owner<world.objects.size();++owner){
                        const auto& object=world.objects[owner];
                        if(object.elementCount!=CELLSu || object.stateCount!=NODESu ||
                           uint64_t(object.elementOffset)+object.elementCount>world.fem.tetrahedra.size())continue;
                        CC_SHA256_CTX hash;CC_SHA256_Init(&hash);
                        uint32_t block[512][5];unsigned filled=0;
                        for(uint32_t i=0;i<object.elementCount;++i){
                            const auto& t=world.fem.tetrahedra[object.elementOffset+i];
                            const auto n=t.nodes;const uint32_t indices[4]={n.x,n.y,n.z,n.w};
                            if(t.identity.y!=owner || !(t.identity.w&NM_OBJECT_ACTIVE)){
                                diagnostics.message="compiled material contains foreign or inactive cells";return diagnostics;
                            }
                            for(unsigned j=0;j<4;++j){
                                if(indices[j]<object.stateOffset || indices[j]-object.stateOffset>=object.stateCount){
                                    diagnostics.message="compiled material contains foreign nodes";return diagnostics;
                                }
                                block[filled][j]=indices[j]-object.stateOffset;
                            }
                            block[filled][4]=t.identity.x;
                            if(++filled==512){CC_SHA256_Update(&hash,block,sizeof(block));filled=0;}
                        }
                        if(filled)CC_SHA256_Update(&hash,block,filled*sizeof(block[0]));
                        unsigned char sum[CC_SHA256_DIGEST_LENGTH];CC_SHA256_Final(sum,&hash);
                        const unsigned char expected[]={BYTES};
                        if(std::memcmp(sum,expected,sizeof(sum))!=0){
                            diagnostics.message="compiled physical skin/fat cell or material disagrees with the registered source";return diagnostics;
                        }
                        if(registeredVolumeOwner!=NM_INVALID_INDEX){diagnostics.message="multiple registered tissue owners";return diagnostics;}
                        registeredVolumeOwner=owner;
                    }
                }
                '''
    check=check.replace('ELEMENTS',src['unchanged_topology']).replace('CELLS',str(len(table))).replace('NODES',str(num)).replace('BYTES',','.join(str(v) for v in bytes.fromhex(promised)))
    out.mkdir();app=out/'app';app.mkdir()
    for n in ('bed_response.inc',):os.link(original/n,out/n)
    for name in ['runtime.mm','app-runtime.mm']:
        s=(original/name).read_text();anchor='                if(rebuilt){\n                    for(uint32_t owner=0;'
        s=substitute(s,anchor,'                '+check+'else if(rebuilt){\n                    for(uint32_t owner=0;')
        key='                if(rebuilt)candidate->executionFingerprint='
        s=substitute(s,key,'                if(certifiedMaterial)candidate->executionFingerprint=mixFingerprint(candidate->executionFingerprint,0x'+promised[:16]+'ull);\n'+key)
        (out/name).write_text(s)
    # Runtime .o is linked both into the scene and into the Matter dynamic library.
    commands=rd(original/'build.json')['commands'][:2]+[before['commands'][3],before['commands'][5]]
    def translated(c):
        pairs=[(original/'app-runtime.mm',out/'app-runtime.mm'),(original/'runtime.mm',out/'runtime.mm'),(base/'libmetalrobo.dylib',out/'libmetalrobo.dylib'),(base/'app/numi-human-native',app/'numi-human-native')]
        return [next((str(b)+v[len(str(a)):] for a,b in pairs if v.startswith(str(a)) and not (v.endswith('numi-human-native.o'))),v) for v in c]
    commands=[translated(c) for c in commands]
    if not all(str(out/k) in commands[i] for i,k in [(2,'runtime.mm.o'),(3,'app-runtime.mm.o')]):raise ValueError('Missing patched runtime link')
    def execute(i):
        with (out/f'build-{i}.log').open('w') as log:subprocess.run(commands[i],stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(execute,[0,1]))
    execute(2);execute(3)
    (out/'metal').mkdir();os.link(base/'metal/NumiMatter.metallib',out/'metal/NumiMatter.metallib')
    info=dict(commands=commands,binary=digest(app/'numi-human-native'),library=digest(out/'libmetalrobo.dylib'),metal=before['metal'],source_elements=src['unchanged_topology'],source_rest=src['material'],source_initial=src['initial'],ordered_nodes_and_material=promised,nodes=num,tetrahedra=len(table))
    (out/'build.json').write_text(json.dumps(info,indent=2)+'\n')
if __name__=='__main__':built(*(Path(v).resolve() for v in sys.argv[1:]))
