"""Transfer one registered fascia attachment band to the existing coupled owner."""
import difflib
import json
import os
from pathlib import Path
import subprocess

root=Path.cwd()
base=root.parent/'resting-clearance-and-atrium-011/registered-first-touch-build'
build=root/'coupled-pectoral-build'
build.mkdir()
name='numilab_human_myosim_visual_probe.mm'
original=(base/name).read_text()
updates=(
    ('    fasciaObjectFirstIndex=static_cast<std::uint32_t>(source.objects.size());',
     '''    constexpr std::uint32_t kCoupledPectoralFasciaObject=3u;
    const bool coupledPectoralFascia =
        std::getenv("NUMI_HUMAN_RESTING_COUPLED_PECTORAL_TRIAL") != nullptr;
    fasciaObjectFirstIndex=static_cast<std::uint32_t>(source.objects.size());'''),
    ('''    for(auto& fasciaObject:fasciaObjects)
        source.objects.push_back(std::move(fasciaObject));''',
     '''    for(auto& fasciaObject:fasciaObjects)
        source.objects.push_back(std::move(fasciaObject));
    if(coupledPectoralFascia) {
        auto& object=source.objects[fasciaObjectFirstIndex+kCoupledPectoralFasciaObject];
        const auto& rows=fasciaAnchorRows[kCoupledPectoralFasciaObject];
        require(!rows.empty() && rows.size()==object.femFixedNodes.size(),
            "coupled pectoral trial requires the complete original anchor band");
        for(const auto& [node,anchor]:rows) {
            require(std::ranges::find(object.femFixedNodes,node)!=object.femFixedNodes.end(),
                "coupled pectoral trial source anchor lacks original ownership");
            numi::matter::FEMHumanAttachmentSource row;
            row.node=node;
            row.bodyIndex=anchor.bodyIndex;
            row.stableIdentifier=0x5fa00000u+node+1u;
            row.localPoint={anchor.localPoint.x,anchor.localPoint.y,anchor.localPoint.z};
            object.femHumanAttachments.push_back(row);
        }
        object.femFixedNodes.clear();
        std::cout << "resting_pectoral_coupled_attachment_trial rows=" << rows.size()
                  << " duplicate_external_rows=0" << std::endl;
    }'''),
    ('''        validateExternalStaticAnchors(fasciaAnchorRows[objectIndex],
            "fascia_"+std::to_string(objectIndex),
            descriptor.stateOffset,descriptor.stateCount);''',
     '''        if(coupledPectoralFascia && objectIndex==kCoupledPectoralFasciaObject) {
            for(const auto& [node,anchor]:fasciaAnchorRows[objectIndex]) {
                const auto& cooked=compiled.world.fem.nodes[descriptor.stateOffset+node];
                require(cooked.restAndFixed.w==2.0f && cooked.positionAndMass.w>0.0f,
                    "coupled pectoral trial lost attachment ownership or mass");
            }
        } else {
            validateExternalStaticAnchors(fasciaAnchorRows[objectIndex],
                "fascia_"+std::to_string(objectIndex),
                descriptor.stateOffset,descriptor.stateCount);
        }'''),
    ('''        for(const auto& [localNode,anchor]:fasciaAnchorRows[objectIndex]) {
            require(localNode<descriptor.stateCount,
                    "registered fascia anchor node is outside its compiled object");''',
     '''        if(coupledPectoralFascia && objectIndex==kCoupledPectoralFasciaObject)
            continue; // Matter owns both kinematics and reaction for this entire band.
        for(const auto& [localNode,anchor]:fasciaAnchorRows[objectIndex]) {
            require(localNode<descriptor.stateCount,
                    "registered fascia anchor node is outside its compiled object");'''),
)
updated=original
for before,after in updates:
    assert updated.count(before)==1, before
    updated=updated.replace(before,after)
(build/name).write_text(updated)
(root/'coupled-pectoral-owner.patch').write_text(''.join(difflib.unified_diff(
    original.splitlines(keepends=True),updated.splitlines(keepends=True),
    'a/apps/'+name,'b/apps/'+name)))
for path in (*base.glob('*.hpp'),base/'runtime.mm.o'):
    os.link(path,build/path.name)
os.symlink(base/'metalrobo',build/'metalrobo',target_is_directory=True)
commands=json.loads((base/'build-commands.json').read_text())
commands=[[x.replace(str(base)+'/',str(build)+'/') for x in command] for command in commands]
(build/'build-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
with (root/'coupled-pectoral-build.log').open('wb') as log:
    for command in commands:
        subprocess.run(command,cwd=Path.home()/'Work/skin-fat-native-integration-209/build-native',
                       stdout=log,stderr=subprocess.STDOUT,check=True)
