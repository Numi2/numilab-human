"""Build and pin the explicit forearm subdivision in the existing native owner."""
import json,sys,subprocess,shutil
from pathlib import Path
from registered_forearms import read,sha
def change(s,old,new,n=1):
    if s.count(old)!=n:raise ValueError('Unexpected owner statement '+old[:100])
    return s.replace(old,new)
def install(source,old,build,out,code):
    args=read(old/'build-commands.json')
    cfg=read(source/'launch.json')
    if any(sha(old/name)!=digest for name,digest in read(old/'identities.json').items()):raise ValueError('Different parent binary')
    build.mkdir();out.mkdir()
    for f in old.iterdir():subprocess.run(['cp','-cR',f,build/f.name],check=True)
    h='NumiHumanForearmBedSections.hpp';shutil.copyfile(code/h,build/h)
    n='numilab_human_myosim_visual_probe.mm';f=build/n;s=f.read_text()
    s=change(s,'MetalMujocoVisualQueries makeMetalMujocoVisualQueries(', '#include "'+h+'"\n\nMetalMujocoVisualQueries makeMetalMujocoVisualQueries(')
    s=change(s,'{20u, 20u, 20u, 30u, 30u, 30u};','{20u, 20u, 20u, 30u, 30u, 30u};\n            const auto selectedParents = forearmBedSectionsRequested() ?\n                std::array<std::uint32_t,6>{12,12,12,22,22,22} : parentRows;')
    s=change(s,'const std::uint32_t parent : parentRows','const std::uint32_t parent : selectedParents')
    s=change(s,'const bool deriveFootManifold =\n        (footManifoldSetting','const bool flatForearmSections=forearmBedSectionsRequested();\n    const bool deriveFootManifold = flatForearmSections ||\n        (footManifoldSetting')
    s=change(s,'require(!deriveFootManifold || (supportGeometryProgram != nullptr &&\n                supportGeometryProgram->valid() &&\n                supportGeometryProgram->usePerContactSupportPlanes),','''require(!deriveFootManifold || (supportGeometryProgram != nullptr &&
                supportGeometryProgram->valid() &&
                (flatForearmSections ? (!supportGeometryProgram->usePerContactSupportPlanes &&
                    useWeightedSkinSupport && !useFEMSkinSupport && restingReleaseInitialization &&
                    !ropeModeRequested && localBedSections.size()==38u)
                  : supportGeometryProgram->usePerContactSupportPlanes)),''')
    f.write_text(s)
    f=build/'NumiHumanRestingVisual.hpp';s=f.read_text()
    s=change(s,'const bool deriveFootManifold=(footManifoldSetting&&','const bool splitArms=forearmBedSectionsRequested();\n        const bool deriveFootManifold=splitArms || (footManifoldSetting&&')
    i='        constexpr unsigned sourceRegionCount=32u;'
    s=change(s,i,i+'''
        if(splitArms) {
            require(localBedLayersRequested() && regionalBedThicknessRequested() && !ropeConfig &&
                    !bedSurface && !organSupportActive && !std::getenv("NUMI_HUMAN_SEPARATION_STEP") &&
                    (!footManifoldSetting || !std::strcmp(footManifoldSetting,"0")),
                    "forearm sections require the registered flat material bed");
            for(unsigned side=0;side<forearmBedParents.size();++side) {
                const auto& row=supportQueries.supportContacts.at(forearmBedParents[side]);
                require(row.sourceGeometryIndex==forearmBedSeedIds[side] && row.bodyIndex==forearmBedBodies[side],
                        "forearm material seed or anatomical owner changed");
            }
        }''')
    s=change(s,'constexpr std::array<unsigned,2> manifoldParents={20u,30u};','const auto manifoldParents=splitArms? forearmBedParents : std::array<unsigned,2>{20u,30u};')
    s=change(s,'if(deriveFootManifold) {\n            for(unsigned parentIndex','if(deriveFootManifold && !splitArms) {\n            for(unsigned parentIndex')
    k='            for (unsigned v=0; v<bedAreas.size(); ++v) {'
    s=change(s,k,'''            const auto materialRow = [&](unsigned r) {
                return splitArms && r>=32u ? regions.at(r).reserved1 : r;
            };
            if(splitArms)partitionForearmBedColumns(posedSkin,bedAreas,seeds,regionForVertex,regionCounts);
'''+k)
    s=change(s,'samples[regionForVertex[vertex]].push_back(i);','samples[materialRow(regionForVertex[vertex])].push_back(i);')
    s=change(s,'samples[row]', 'samples[materialRow(row)]',3)
    s=change(s,'<<(deriveFootManifold?(organSupportActive?', '<<(splitArms?"source_footprint_four_patches_per_forearm":deriveFootManifold?(organSupportActive?')
    f.write_text(s)
    base=Path(next(args[2][i+1] for i,x in enumerate(args[2]) if x=='-iquote' and (Path(args[2][i+1])/'NumiHumanRestingSupportGeometry.hpp').exists()))
    f=build/'NumiHumanRestingSupportGeometry.hpp';s=(base/f.name).read_text()
    s=change(s,'(parent == 20u || parent == 30u) && parent < r &&','''parent < r && (parent == 20u || parent == 30u ||
                            (parent==12u && contacts[parent].sourceGeometryIndex==6890u && contacts[parent].bodyIndex==42u) ||
                            (parent==22u && contacts[parent].sourceGeometryIndex==47596u && contacts[parent].bodyIndex==92u)) &&''')
    f.write_text(s)
    cmds=[[v.replace(str(old)+'/',str(build)+'/') for v in row] for row in args]
    runtime=Path(cfg['environment']['DYLD_LIBRARY_PATH'])/'libmetalrobo.dylib'
    if sha(runtime)!=cfg['pins'][str(runtime)]:raise ValueError('Different physical backend')
    for cmd in cmds[2:]:subprocess.run(cmd,cwd=runtime.parent.parent,check=True)
    def save(p,j):p.write_text(json.dumps(j,indent=2)+'\n')
    save(build/'build-commands.json',cmds)
    names=set(read(old/'identities.json'))|{h,f.name,n,"NumiHumanRestingVisual.hpp"}
    save(build/'identities.json',{n:sha(build/n) for n in sorted(names)})
    cfg['environment']['NUMI_HUMAN_FOREARM_BED_SECTIONS']='1'
    cfg['parent_launch']=sha(source/'launch.json')
    save(out/'launch.json',cfg)
if __name__=='__main__':install(*(Path(a).resolve() for a in sys.argv[1:]))
