"""Extend the registered native contact manifold with two four-patch pelvic regions."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def build(base,dest,launch):
    identities=json.loads((base/'identities.json').read_bytes())
    if any(sha(base/k)!=v for k,v in identities.items()):raise ValueError('Native parent changed')
    if dest.exists():raise FileExistsError(dest)
    subprocess.run(['cp','-cR',base,dest],check=True)
    target='NumiHumanPelvicBedSections.hpp'
    shutil.copy2(Path(__file__).with_name(target),dest/target)
    def patch(name,changes):
        p=dest/name; text=p.read_text()
        for old,new,count in changes:
            if text.count(old)!=count:raise ValueError((name,old[:80],text.count(old)))
            text=text.replace(old,new)
        p.write_text(text)
    config='numiPelvicBed::enabled()'
    patch('numilab_human_myosim_visual_probe.mm',[
        ('#include "NumiHumanForearmBedSections.hpp"','#include "NumiHumanForearmBedSections.hpp"\n#include "'+target+'"',1),
        ('(deriveFootManifold ? 6u : 0u)','(deriveFootManifold ? ('+config+' ? 12u : 6u) : 0u)',5),
        ('const auto selectedParents = forearmBedSectionsRequested() ?', 'const auto baseParents = forearmBedSectionsRequested() ?',1),
        ('            require(support->records.size() == sourceRows &&\n                        result.supportContacts.size() == sourceRows,','''            std::vector<unsigned> selectedParents(baseParents.begin(),baseParents.end());
            if(numiPelvicBed::enabled()) {
                require(forearmBedSectionsRequested(),"pelvis subdivision requires the registered 38-row parent");
                for(auto seed:numiPelvicBed::seeds) selectedParents.insert(selectedParents.end(),3,seed.parent);
            }
            require(support->records.size() == sourceRows &&
                        result.supportContacts.size() == sourceRows,''',1),
        ('result.points.size() + parentRows.size()', 'result.points.size() + selectedParents.size()',1),
        ('localBedSections.size()==38u','localBedSections.size()==('+config+' ? 44u : 38u)',1)])
    patch('NumiHumanForearmBedSections.hpp',[
        ('counts.size()!=38u','(counts.size()!=38u && counts.size()!=44u)',1)])
    patch('NumiHumanRestingVisual.hpp',[
        ('const unsigned footDerivedCount=deriveFootManifold?6u:0u;','''const bool splitPelvis=numiPelvicBed::enabled();
        require(!splitPelvis || splitArms,"pelvic sections require the registered flat-bed layout");
        if(splitPelvis) for(auto seed:numiPelvicBed::seeds) {
            const auto& row=supportQueries.supportContacts.at(seed.parent);
            require(row.bodyIndex==seed.body && row.sourceGeometryIndex==seed.geometry,"pelvic NHCNT identity mismatch");
        }
        const unsigned footDerivedCount=deriveFootManifold?(splitPelvis?12u:6u):0u;''',1),
        ('derived<6u;++derived','derived<footDerivedCount;++derived',1),
        ('const unsigned parent=derived<3u?manifoldParents[0]:manifoldParents[1];','const unsigned parent=derived<6u?manifoldParents[derived/3]:numiPelvicBed::seeds[(derived-6u)/3u].parent;',1),
        ('            if(splitArms)partitionForearmBedColumns(posedSkin,bedAreas,seeds,regionForVertex,regionCounts);','''            if(splitArms)partitionForearmBedColumns(posedSkin,bedAreas,seeds,regionForVertex,regionCounts);
            if(splitPelvis)numiPelvicBed::partition(posedSkin,bedAreas,seeds,regionForVertex,regionCounts);''',1),
        ('<<(splitArms?"source_footprint_four_patches_per_forearm"', '<<(splitPelvis?"source_footprint_four_patches_per_forearm_and_pelvis":splitArms?"source_footprint_four_patches_per_forearm"',1)])
    patch('NumiHumanRestingSupportGeometry.hpp',[
        ('parent == 20u || parent == 30u ||','''parent == 20u || parent == 30u ||
                            (numiPelvicBed::enabled() && std::any_of(numiPelvicBed::seeds.begin(),numiPelvicBed::seeds.end(),[&](auto seed) {
                                return parent==seed.parent && contacts[parent].sourceGeometryIndex==seed.geometry &&
                                       contacts[parent].bodyIndex==seed.body;
                            })) ||''',1)])
    patch('NumiHumanRestingSupportGeometry.hpp',[('#pragma once','#pragma once\n#include "'+target+'"',1)])
    cfg=json.loads(launch.read_bytes());runtime=Path(cfg['environment']['DYLD_LIBRARY_PATH'])/'libmetalrobo.dylib'
    if sha(runtime)!=cfg['pins'][str(runtime)]:raise ValueError('Native runtime changed')
    cmds=[[v.replace(str(base)+'/',str(dest)+'/') for v in cmd] for cmd in json.loads((base/'build-commands.json').read_bytes())]
    with (dest/'build.log').open('w')as f:
        for cmd in cmds[2:]:subprocess.run(cmd,cwd=runtime.parent.parent,stdout=f,stderr=subprocess.STDOUT,check=True)
    (dest/'build-commands.json').write_text(json.dumps(cmds,indent=2)+'\n')
    (dest/'identities.json').write_text(json.dumps({k:sha(dest/k) for k in sorted(set(identities)|{target})},indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('base','dest','launch'):p.add_argument(k,type=lambda s:Path(s).resolve())
    build(**vars(p.parse_args()))
