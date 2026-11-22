"""Reuse registered ligament material coordinates, with an independent current-pose clearance gate."""
from pathlib import Path
import json,shutil,subprocess,sys
from build_distinct_registry import replace
from evaluate_registered_bed_transfer import digest
src,app,out=map(lambda p:Path(p).resolve(),sys.argv[1:])
prior=json.loads((src/'identities.json').read_text());main='numilab_human_myosim_visual_probe.mm'
if digest(src/main)!=prior['source'] or digest(src/'numi-human-native')!=prior['binary']:raise ValueError('changed full native owner')
out.mkdir(exist_ok=False)
for f in src.iterdir():
    if f.is_file() and f.suffix in ('.hpp','.metal','.metalinc','.metallib','.inc'):(out/f.name).hardlink_to(f)
(out/'metalrobo').symlink_to((src/'metalrobo').resolve(),target_is_directory=True)
header=Path(__file__).resolve().parent/'NumiHumanLoadedLigamentPose.hpp';shutil.copyfile(header,out/header.name)
s=(src/main).read_text()
s=replace(s,'#include "NumiHumanLoadedSkinPose.hpp"','#include "NumiHumanLoadedSkinPose.hpp"\n#include "NumiHumanLoadedLigamentPose.hpp"')
s=replace(s,'        registeredLigaments.reserve(catalog.size());','''        std::vector<MRBodyStateGPU> registeredReferenceBodies;
        std::vector<metalrobo::inferred_acl::BoneSurfaceStorage> materialStorage;
        std::vector<metalrobo::inferred_acl::BoneSurface> materialSurfaces;
        if(numiLoadedSkin::enabled()){
            registeredReferenceBodies=bodies;numiLoadedSkin::sourceFrames(registeredReferenceBodies);
            materialStorage.reserve(storage.size());materialSurfaces.reserve(storage.size());
            for(const auto& existing:surfaces){
                const auto it=std::find_if(bones->records.begin(),bones->records.end(),[&](const auto& rec){
                    return rec.bodyIndex==existing.body&&rec.stableId==existing.stableId;});
                require(it!=bones->records.end(),"registered ligament reference surface missing");
                const auto i=size_t(it-bones->records.begin());
                materialStorage.push_back(metalrobo::inferred_acl::bakeRegisteredBoneSurface(*it,bones->sourceRecordIndices[i],
                    std::span<const BoneVertex>(bones->vertices),std::span<const std::uint32_t>(bones->indices),
                    registeredReferenceBodies.at(existing.body),bones->payloadSha256));
                materialSurfaces.push_back(materialStorage.back().view());
            }
        }
        registeredLigaments.reserve(catalog.size());''')
s=replace(s,'''                registeredLigaments.push_back(
                    metalrobo::registered_ligaments::cookReference(
                        spec,model,bodies,surfaces,sourceBodyMap,ownership));''','''                const bool loaded=numiLoadedSkin::enabled();
                auto fiber=metalrobo::registered_ligaments::cookReference(spec,model,loaded?registeredReferenceBodies:bodies,
                    loaded?materialSurfaces:surfaces,sourceBodyMap,ownership);
                if(loaded)certifyLoadedLigament(fiber,registeredReferenceBodies,bodies,materialSurfaces,surfaces);
                registeredLigaments.push_back(std::move(fiber));''')
(out/main).write_text(s)
cmds=[[a.replace(str(src)+'/',str(out)+'/') for a in c] for c in prior['commands']]
for cmd in cmds:subprocess.run(cmd,check=True,cwd=app/'build-native')
(out/'identities.json').write_text(json.dumps(dict(parent=prior['binary'],commands=cmds,source=digest(out/main),binary=digest(out/'numi-human-native'),header=digest(header)),indent=2)+'\n')
