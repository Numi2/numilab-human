"""Compile disjoint surface quadratures into the existing single nodal contact operator."""
from pathlib import Path
import sys,json,subprocess,os,shutil
from concurrent.futures import ThreadPoolExecutor
from compile_connected_joint import digest

def substitute(text,old,new,count=1):
    if text.count(old)!=count:raise ValueError('Unmatched owner source: '+old[:100])
    return text.replace(old,new)

def run(job,base,out):
    runtime=job/'integrated';owner=Path('/Users/numi/Work/skin-fat-native-integration-209/matter/include')
    inc=out/'include';header=Path('numi/matter/fem_body_triangle_contact_sample.hpp')
    app=out/'app';metal=out/'metal';h=inc/header
    for folder in [h.parent,app,metal/'src/metal']:folder.mkdir(parents=True,exist_ok=True)
    # A negative Float32 bit pattern identifies an explicit constituent peak
    # stiffness. Legacy rows keep a zero extension. Unmarked bits fail admission.
    text=(owner/header).read_text()
    text=substitute(text,'#include <cmath>','#include <cmath>\n#include <bit>')
    anchor='inline bool validTriangleContactSample('
    text=substitute(text,anchor,'''inline float trianglePeakStiffness(const NMNumiHumanFEMBodyContactSampleGPU& sample) {
    if (sample.reserved0 == 0u) return sample.bodyLocalNormalAndReferenceSeparation.w;
    return -std::bit_cast<float>(sample.reserved0);
}

'''+anchor)
    text=substitute(text,'sample.flags != requiredFlags || sample.reserved0 != 0u','sample.flags != requiredFlags')
    text=substitute(text,'    std::array<double,3> unitNormal{};', '''    const float peak=trianglePeakStiffness(sample);
    if (!std::isfinite(peak) || !(peak>=p1.w)) return false;
    std::array<double,3> unitNormal{};''')
    h.write_text(text)
    for f in (base/'app').iterdir():
        if f.suffix in ['.hpp','.inc','.mm','.h'] and f.suffix!='.mm':os.link(f.resolve(),app/f.name)
    main='numilab_human_myosim_visual_probe.mm';text=(base/'app'/main).read_text()
    key='            std::map<std::pair<unsigned,unsigned>,const Triangle*> corticalSeeds;'
    text=substitute(text,key,'''            // Only the O/D templates from this mesh and its explicitly registered,
            // disjoint cortical faces may combine. Fascia rows and other meshes keep
            // their ordinary distinct-owner admission.
            std::unordered_map<unsigned,std::size_t> tissueRows;
            std::unordered_map<unsigned,unsigned> materialMask, corticalMask;
            const auto triangleFlags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
                                     NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT;
            for (std::size_t i=0;i<fasciaBodyContactSamples.size();++i) {
                const auto& s=fasciaBodyContactSamples[i];
                if(s.flags!=triangleFlags || s.slaveNode<descriptor.stateOffset ||
                   s.slaveNode-descriptor.stateOffset>=descriptor.stateCount) continue;
                require(tissueRows.emplace(s.slaveNode,i).second,"overlapping baseline skin contact");
            }
            std::size_t joined=0,mixed=0;
'''+key)
    old='                fasciaBodyContactSamples.push_back(sample);fasciaBodyContactNeighborhoodRanges.push_back({});'
    text=substitute(text,old,'''                const unsigned id=unsigned(node), materialBit=1u<<(row.material-15u);
                require((corticalMask[id]&materialBit)==0,"duplicate registered cortical material quadrature");
                corticalMask[id]|=materialBit;
                const auto [position,inserted]=tissueRows.emplace(id,fasciaBodyContactSamples.size());
                if(inserted){
                    fasciaBodyContactSamples.push_back(sample);
                    fasciaBodyContactNeighborhoodRanges.push_back({});
                    materialMask[id]=materialBit;
                    continue;
                }
                auto& existing=fasciaBodyContactSamples[position->second];
                require(triangleContact::validTriangleContactSample(existing),"invalid incident material contact");
                const double a0=existing.bodyLocalPointAndArea.w,a1=sample.bodyLocalPointAndArea.w;
                const double k0=existing.bodyLocalNormalAndReferenceSeparation.w,k1=sample.bodyLocalNormalAndReferenceSeparation.w;
                const double c0=existing.stiffnessAndNormalStrainPerPressure.w,c1=sample.stiffnessAndNormalStrainPerPressure.w;
                const auto peak=std::max(triangleContact::trianglePeakStiffness(existing),float(k1));
                const auto area=float(a0+a1);
                const auto stiffness=float((a0*k0+a1*k1)/(a0+a1));
                existing.bodyLocalPointAndArea.w=area;
                existing.bodyLocalNormalAndReferenceSeparation.w=stiffness;
                existing.stiffnessAndNormalStrainPerPressure.w=float(std::max(k0*c0,k1*c1)/stiffness);
                existing.reserved0=peak==stiffness?0u:std::bit_cast<unsigned>(-peak);
                require(triangleContact::validTriangleContactSample(existing),"invalid combined material contact");
                ++joined;
                if(c0!=c1)++mixed;
                materialMask[id]|=materialBit;''')
    text=substitute(text,'<<" nodal_rows="<<std::size(registeredCorticalSkin::samples)', '<<" combined_incident_rows="<<joined<<" different_compliance_rows="<<mixed<<" nodal_rows="<<std::size(registeredCorticalSkin::samples)')
    # Limit baseline scope explicitly to outer/deep templates produced by this loop.
    marker='        for(unsigned layer=0u;layer<2u;++layer) {\n            const auto points=layer==0u'
    text=substitute(text,marker,'        const std::size_t firstRegisteredTissueRow=fasciaBodyContactSamples.size();\n'+marker)
    text=substitute(text,'std::size_t i=0;i<fasciaBodyContactSamples.size();++i) {\n                const auto& s=','std::size_t i=firstRegisteredTissueRow;i<fasciaBodyContactSamples.size();++i) {\n                const auto& s=')
    text=text.replace('materialMask, corticalMask','corticalMask').replace('                    materialMask[id]=materialBit;\n','').replace('                materialMask[id]|=materialBit;','')
    (app/main).write_text(text)
    for f in (runtime/'metal/src').iterdir():
        if f.name!='metal':
            if f.name=='matter_kernels.metal':shutil.copy2(f,metal/'src'/f.name)
            else:(metal/'src'/f.name).symlink_to(f.resolve())
    for f in (runtime/'metal/src/metal').iterdir():
        if f.name!='numi_human.metalinc':(metal/'src/metal'/f.name).symlink_to(f.resolve())
    f=Path('src/metal/numi_human.metalinc');text=(runtime/'metal'/f).read_text()
    mk='inline bool finite4('
    text=substitute(text,mk,'''inline float humanTrianglePeakStiffness(NMNumiHumanFEMBodyContactSampleGPU sample) {
    return sample.reserved0==0u ? sample.bodyLocalNormalAndReferenceSeparation.w : -as_type<float>(sample.reserved0);
}
inline bool humanTriangleHasValidPeak(NMNumiHumanFEMBodyContactSampleGPU sample) {
    const float k=humanTrianglePeakStiffness(sample);
    return isfinite(k) && k>=sample.bodyLocalNormalAndReferenceSeparation.w;
}
'''+mk)
    text=substitute(text,'sample.reserved0 != 0u || !finite4(packed0)','!humanTriangleHasValidPeak(sample) || !finite4(packed0)')
    text=substitute(text,'sample.reserved0!=0u || !finite4(sample.bodyLocalPointAndArea)','!humanTriangleHasValidPeak(sample) || !finite4(sample.bodyLocalPointAndArea)')
    text=substitute(text,'maximumPressure = max(maximumPressure, pressure);\n        storedEnergy += 0.5f * pressure * closure *\n            contactAreaForSample;', '''maximumPressure = max(maximumPressure,triangle ? humanTrianglePeakStiffness(sample)*closure : pressure);
        storedEnergy += 0.5f * pressure * closure *
            contactAreaForSample;''',2)
    (metal/f).write_text(text)
    host=runtime/'numi_human.mm';text=host.read_text()
    text=substitute(text,'sample.reserved0 != 0u || !validPayload ||','(!triangleContact && sample.reserved0 != 0u) || !validPayload ||')
    (out/host.name).write_text(text)
    def replaced(c,subs):
        for a,b in subs:
            c=[w.replace(str(a),str(b)) for w in c]
        return c
    cmds=json.loads((runtime/'build.json').read_bytes())['commands'][:4]+json.loads((base/'build.json').read_bytes())['commands']
    changes=[(runtime/'numi_human.mm',out/'numi_human.mm'),(runtime/'libmetalrobo.dylib',out/'libmetalrobo.dylib'),(runtime/'metal',metal),(base/'app',app)]
    cmds=[replaced(c,changes) for c in cmds]
    for i in [2,4]:cmds[i][1:1]=['-I',str(inc)]
    def execute(indices):
        for i in indices:
            with (out/f'build-{i}.log').open('w') as log:subprocess.run(cmds[i],stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        m=pool.submit(execute,[0,1]);execute([2,3,4,5]);m.result()
    result=dict(commands=cmds,binary=digest(app/'numi-human-native'),library=digest(out/'libmetalrobo.dylib'),metal=digest(metal/'NumiMatter.metallib'),host=digest(out/host.name),native=digest(app/main),triangle=digest(h))
    (out/'build.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
