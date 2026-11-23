"""Build native host admission, Metal force assembly, and the integrated Human application together."""
from pathlib import Path
import json,sys,subprocess,os
from concurrent.futures import ThreadPoolExecutor
from compile_connected_joint import digest
from prepare_live_cortex import exact,write

def build(h,host_base,runtime,out):
    scene=h/'native-scene-build';shader=h/'cortical-metal-body-pose'
    model=out/'app';metal=out/'metal';model.mkdir(exist_ok=True);metal.mkdir(exist_ok=True)
    for f in scene.iterdir():
        if f.suffix in ('.hpp','.inc','.mm') and not (model/f.name).exists():
            (model/f.name).symlink_to(f)
    for f in ['NumiHumanRegisteredLigamentMaterial.inc','NumiHumanCorticalRegistration.inc']:
        write(model/f,(out/f).read_text())
    name='NumiHumanRegisteredLigaments.hpp';s=(scene/name).read_text()
    s=exact(s,'struct Spec {','#include "NumiHumanCorticalRegistration.inc"\nstruct Spec {')
    pos=' const V3 axis='
    s=exact(s,pos,''' const RegisteredCorticalFiber* cortex=nullptr;
 if(material)for(const auto& candidate:registeredCorticalFibers)
    if(candidate.name==s.name && candidate.side==s.side)cortex=&candidate;
 if(cortex){
    // The source 3D certificate references all registered source banks and this
    // exact reference configuration. A new rest pose needs its own certificate.
    if(!map.empty() || poses.size()!=std::size(registeredCorticalPose) ||
       surfaces.size()!=185)throw std::runtime_error("cortical reference ownership mismatch");
    for(const auto& bank:surfaces)if(bank.payloadSha256!=kReferenceBonesPayloadSha256)
        throw std::runtime_error("mixed cortical reference payload");
    for(std::size_t i=0;i<poses.size();++i){
        const auto& b=poses[i];const float current[]={b.position.x,b.position.y,b.position.z,b.orientation.x,b.orientation.y,b.orientation.z,b.orientation.w};
        for(unsigned j=0;j<7;++j)if(current[j]!=registeredCorticalPose[i][j])
            throw std::runtime_error("unverified cortical reference pose at body "+std::to_string(i));
    }
    if(inferred_acl::detail::norm(inferred_acl::detail::sub(p0.p,cortex->start))>2e-7 ||
       inferred_acl::detail::norm(inferred_acl::detail::sub(p1.p,cortex->end))>2e-7 ||
       L>cortex->length+2e-7)
        throw std::runtime_error("cortical path does not join registered material sites");
    L=cortex->length;
 }else{
'''+pos)
    s=exact(s,' const double referenceLength=',' }\n const double referenceLength=')
    s=exact(s,'r.reserved0=0;','r.reserved0=cortex?cortex->tag:0;')
    s=exact(s,'q.limitations=std::string(s.limitations);return out;', '''q.limitations=std::string(s.limitations);
 if(cortex){
    q.geometrySource="Source-bone barycentric material sites and an exterior cortical path, registered to this resting pose.";
    q.limitations=s.group==Group::coracoclavicular ?
        "Inferred effective two-body fiber. Sliding, unilateral cortical contacts carry the span reaction." :
        "MyoSim source attachment and slack-length priors. A feasible exterior path bounds slack tension. Taut excursions require a sliding path.";
 }
 return out;''')
    write(model/name,s)
    # Copy the primary translation unit so its quoted includes resolve in this variant.
    main='numilab_human_myosim_visual_probe.mm';app_source=(scene/main).read_text()
    start=app_source.index('        // Optional source-pose inspection paired')
    finish=app_source.index('        if(const char* candidateDirectory=',start)
    snapshot=app_source[start:finish]
    marker='        const auto skinRestBodies=visualBodyStates(model,skinRestResult.bodyPoses);'
    app_source=exact(exact(app_source,snapshot,''),marker,marker+'\n'+snapshot)
    write(model/main,app_source)
    tree=metal/'src';(tree/'metal').mkdir(parents=True,exist_ok=True)
    for f in (shader/'src').iterdir():
        if f.name=='metal':continue
        dst=tree/f.name
        if f.name=='matter_kernels.metal':write(dst,f.read_text())
        elif not dst.exists():dst.symlink_to(f.resolve())
    for f in (shader/'src/metal').iterdir():
        dst=tree/'metal'/f.name
        if f.name!='numi_human.metalinc':
            if not dst.exists():dst.symlink_to(f.resolve())
            continue
        s=f.read_text()
        for fun in ['nhSlidingLigamentPath','nhSolveCorticalPath']:
            s=exact(s,'inline bool '+fun+'(device const NHCorticalLigamentEdge* edges,','template<class EdgePointer> inline bool '+fun+'(EdgePointer edges,')
        p='inline bool passiveLigamentState('
        s=exact(s,p,(out/'CorticalTables.metalinc').read_text()+(out/'native_free_span.metalinc').read_text()+'\n'+p)
        s=exact(s,'ligament.reserved0 != 0u ||','(ligament.reserved0 != 0u && !nhKnownCorticalRow(ligament)) ||')
        p='    firstPoint = firstPose.position.xyz + humanQuaternionRotate('
        s=exact(s,p,'''    if(ligament.reserved0!=0u)
        return nhRegisteredFreeSpan(ligament,firstPose,secondPose,firstPoint,secondPoint,firstForce,effectiveStretch,tension);
'''+p)
        write(dst,s)
    host=out/'numi_human.mm';s=(host_base/host.name).read_text()
    s=exact(s,'bool evaluateNumiHumanPassiveLigamentFiber(', (out/'CorticalIdentity.inc').read_text()+'\nbool evaluateNumiHumanPassiveLigamentFiber(')
    s=exact(s,'ligament.reserved0 != 0u ||','(ligament.reserved0 != 0u && !nhKnownCorticalRow(ligament)) ||',2)
    write(host,s)
    def cmds(file):return json.loads(file.read_text())['commands']
    def sub(cmd,mapping):
        return [replace_arg(a,mapping) for a in cmd]
    def replace_arg(a,mapping):
        for k,v in mapping.items():a=a.replace(str(k),str(v))
        return a
    hg=json.loads((host_base/'build-commands.json').read_text())
    if isinstance(hg,dict):hg=hg['commands']
    hc,=[c for c in hg if str(host_base/host.name) in c]
    hc=sub(hc,{host_base/host.name:host});hc[1:1]=['-iquote',str(host_base)]
    link=cmds(runtime/'build.json')[2]
    lib=out/'libmetalrobo.dylib'
    link=sub(link,{runtime/'libmetalrobo.dylib':lib,host_base/'numi_human.mm.o':out/'numi_human.mm.o'})
    if str(out/'numi_human.mm.o') not in link:raise ValueError('Host owner not linked')
    sc=[sub(c,{scene:model,runtime/'libmetalrobo.dylib':lib}) for c in cmds(scene/'registration.json')]
    mc=[sub(c,{shader:metal}) for c in cmds(shader/'build.json')]
    def group(name,commands):
        with (out/(name+'.log')).open('wb') as log:
            for c in commands:subprocess.run(c,stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(3) as pool:
        jobs=[pool.submit(group,key,cs) for key,cs in [('metal',mc),('host',[hc,link]),('app',[sc[0]])]]
        for future in jobs:future.result()
    group('link',sc[1:])
    result=dict(commands=mc+[hc,link]+sc,source_geometry=digest(out/'registration.json'),
                inputs={str(f):digest(f) for f in [model/name,model/'NumiHumanRegisteredLigamentMaterial.inc',model/'NumiHumanCorticalRegistration.inc',tree/'metal/numi_human.metalinc',host]},
                binaries={str(f):digest(f) for f in [model/'numi-human-native',lib,metal/'NumiMatter.metallib']})
    (out/'build.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':build(*(Path(x).resolve() for x in sys.argv[1:]))
