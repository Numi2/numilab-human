"""Build the registered supported Human owner with one projected footprint per bed coordinate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def add_registration_receipt(s):
    marker='                for (unsigned v = 0; v < posedSkin.size(); ++v) {\n                    if (tributaryAreas[v] == 0) continue;'
    if s.count(marker)!=1: raise ValueError('Missing registered column loop')
    s=s.replace(marker,'''                std::ofstream footprintReceipt;
                if(std::getenv("NUMI_HUMAN_BED_QUADRATURE_CAPTURE")){
                    footprintReceipt.open(outputDirectory / "resting-local-bed-footprint.csv");
                    require(bool(footprintReceipt),"cannot capture registered footprint");
                    footprintReceipt<<"vertex,row,source_geometry_index,x_m,y_m,z_m,area_m2,skin_m,fat_m\\n"
                        <<std::setprecision(17);
                }
'''+marker)
    marker='                    materialColumns[row].add(tributaryAreas[v],'
    if s.count(marker)!=1: raise ValueError('Missing registered column material')
    s=s.replace(marker,'''                    if(footprintReceipt.is_open()){
                        footprintReceipt<<v<<','<<row<<','<<regions[row].sourceGeometryIndex<<','
                            <<posedSkin[v].x<<','<<posedSkin[v].y<<','<<posedSkin[v].z<<','
                            <<tributaryAreas[v]<<','<<kWholeBodySkinThicknessM[nearest]<<','
                            <<kWholeBodyFatThicknessM[nearest]<<'\\n';
                    }
'''+marker)
    marker='        registrationBodyIndices.insert(anatomyGPU.bodyAndFlags.x);'
    if s.count(marker)!=1: raise ValueError('Missing native initial registration')
    s=s.replace(marker, '''        if (std::getenv("NUMI_HUMAN_BED_QUADRATURE_CAPTURE")) {
            for (unsigned body=0; body<initialBodies.size(); ++body) registrationBodyIndices.insert(body);
        }
'''+marker)
    return s


def build(parent, app, out, capture_registration=False):
    source = Path(__file__).resolve().parent
    ids = json.loads((parent/'identities.json').read_text())
    for file, identity in ids.items():
        if sha(parent/file) != identity:
            raise ValueError(f'Changed native source {file}')
    out.mkdir(exist_ok=False)
    # Clone the working native input tree. Reuse the identical GPU library and
    # compile only the changed authoring code; runtime buffers/layout stay intact.
    subprocess.run(['cp', '-cR', str(parent) + '/.', str(out)], check=True)
    header = out/'NumiHumanBedFootprint.hpp'
    shutil.copyfile(source/header.name, header)
    path = out/'NumiHumanRestingVisual.hpp'
    s = path.read_text()
    anchor = '#include "NumiHumanBedColumnQuadrature.hpp"'
    if s.count(anchor) != 1:
        raise ValueError('Unmatched regional section owner')
    s=s.replace(anchor, anchor+'\n#include "NumiHumanBedFootprint.hpp"')
    begin=s.index('            for (std::size_t face = 0; face < skin.indices.size(); face += 3)',s.index('constexpr float patchDepth ='))
    end=s.index('            std::vector<NumiHumanBedColumnQuadrature>',begin)
    s=s[:begin]+'''            std::vector<numiHumanBedFootprint::Point> footprintPoints;
            std::vector<double> heightLimit;
            footprintPoints.reserve(posedSkin.size());
            heightLimit.reserve(posedSkin.size());
            for (unsigned v=0; v<posedSkin.size(); ++v) {
                const auto& p=posedSkin[v];
                footprintPoints.push_back({p.x,p.y,p.z});
                heightLimit.push_back(double(patchMinimum.at(regionForVertex[v]))+patchDepth);
            }
            const auto bedAreas=numiHumanBedFootprint::areas(footprintPoints,skin.indices,heightLimit);
            for (unsigned v=0; v<bedAreas.size(); ++v) {
                areas.at(regionForVertex[v])+=bedAreas[v];
                if (useRegionalThickness) tributaryAreas[v]=bedAreas[v];
            }
            std::cout<<"resting_bed_footprint=lowest_projected_skin triangles="<<skin.indices.size()/3u
                     <<" vertices="<<bedAreas.size()<<" projected_area_m2="
                     <<std::accumulate(areas.begin(),areas.end(),0.)<<std::endl;
'''+s[end:]
    if capture_registration:
        s = add_registration_receipt(s)
    path.write_text(s)
    commands=[[v.replace(str(parent)+'/',str(out)+'/') for v in cmd]
              for cmd in json.loads((parent/'build-commands.json').read_text())]
    (out/'build-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    for cmd in commands:
        if 'clang++' in cmd or any(v.endswith('clang++') for v in cmd):
            subprocess.run(cmd,cwd=app/'build-native',check=True)
    if sha(out/'numi-human-native')==ids['numi-human-native']:
        raise ValueError('Changed code did not enter the native executable')
    ids.update({f:sha(out/f) for f in (path.name,header.name,'numi-human-native')})
    (out/'identities.json').write_text(json.dumps(ids,indent=2)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent','app','out'):
        parser.add_argument(name, type=lambda p: Path(p).resolve())
    parser.add_argument('--with-registration-capture', action='store_true')
    args = parser.parse_args()
    build(args.parent, args.app, args.out, args.with_registration_capture)
