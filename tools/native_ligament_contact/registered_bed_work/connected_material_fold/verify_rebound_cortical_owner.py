from pathlib import Path
import json,subprocess
from compile_connected_joint import digest
def run(q):
    out=q/'registered-material-contacts';v=q/'qualified-material'
    r=next(a for a in q.parents if a.name=='physical-shoulder-registration-066');parent=r/'motion-repair-work/native-material'
    record=json.loads((parent/'native-import-build.json').read_bytes());cmd=record['command'];source=Path(cmd[cmd.index('-o')-1]);original=source.read_text()
    if digest(source)!=record['source']:raise ValueError('Unverified native importer source')
    inc=q/'gravity-contact/app';test=out/'verify-rebound.cpp';exe=out/'verify-rebound'
    pos=original.index('int main(')
    header='#include "NumiHumanUnpairedSkinFatVolume.hpp"\n#include "NumiHumanCorticalSkinBinding.inc"\n'
    original=original[:pos]+header+original[pos:]
    before='    const auto mass = read_masses'
    body='''    if (imported.nodeFileSHA256 != registeredCorticalSkin::material_reference ||
            imported.elementFileSHA256 != registeredCorticalSkin::elements)
        throw std::runtime_error("Cortical contact belongs to different material geometry");
    const auto cortical = std::span(registeredCorticalSkin::faces);
    std::uint64_t priorSlot = 0;
    bool first = true;
    std::array<double,2> faceArea{}, sampleArea{};
    for (const auto& face : cortical) {
        if (face.child >= tets.size() || face.side >= 4 || (!first && face.slot() <= priorSlot))
            throw std::runtime_error("Invalid cortical slot order");
        priorSlot = face.slot();
        first = false;
        const auto& tet = tets[face.child];
        std::array<std::uint32_t,3> triangle{};
        for (unsigned j=0,k=0;j<4;++j) if (j!=face.side) triangle[k++]=tet[j];
        if (tissue::corticalFacetAt(cortical,face.child,face.side)!=&face ||
            !tissue::corticalFacetMatches(face,triangle,bundle.childMaterialIndices[face.child]))
            throw std::runtime_error("Physical cell side mismatch");
        double a[3],b[3];
        for (unsigned j=0;j<3;++j) {
            a[j]=points[triangle[1]][j]-points[triangle[0]][j];
            b[j]=points[triangle[2]][j]-points[triangle[0]][j];
        }
        faceArea.at(face.material-15)+=.5*std::hypot(a[1]*b[2]-a[2]*b[1],
                a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]);
    }
    for (const auto& sample:registeredCorticalSkin::samples) {
        if(sample.node>=points.size() || !(sample.area>0) || !(sample.thickness>0) ||
           !std::isfinite(sample.area) || !std::isfinite(sample.thickness)) throw std::runtime_error("Invalid contact sample");
        sampleArea.at(sample.material-15)+=sample.area;
    }
    for (unsigned j=0;j<2;++j)
        if (std::abs(faceArea[j]-sampleArea[j])>1e-12*faceArea[j]) throw std::runtime_error("Lost cortical contact area");
    std::cerr<<"native cortical faces "<<cortical.size()<<" samples "<<std::size(registeredCorticalSkin::samples)
        <<" skin_area_m2 "<<faceArea[0]<<" fat_area_m2 "<<faceArea[1]<<"\\n";
'''
    if original.count(before)!=1:raise ValueError('Native importer drift')
    test.write_text(original.replace(before,body+before))
    cmd=[cmd[0],'-iquote',str(out/'app'),'-iquote',str(inc),*cmd[1:]];cmd[cmd.index('-o')-1]=str(test);cmd[cmd.index('-o')+1]=str(exe)
    with (out/'build.log').open('w')as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    run=json.loads((v/'result.json').read_bytes())['native_command'];run[0]=str(exe)
    with (out/'native-result.json').open('w')as f,(out/'native-contacts.log').open('w')as err:p=subprocess.run(run,stdout=f,stderr=err)
    (out/'native-check.json').write_text(json.dumps(dict(exit=p.returncode,build=cmd,command=run,source=digest(test),binary=digest(exe),program=digest(out/'app/NumiHumanCorticalSkinBinding.inc'),report=digest(out/'native-result.json')),indent=2)+'\n')
    if p.returncode:raise RuntimeError((out/'native-contacts.log').read_text())
if __name__=='__main__':run(Path(__file__).resolve().parent)
