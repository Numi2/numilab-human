"""Expose the limiting anatomical coordinates from the existing initial equilibrium result."""
from pathlib import Path
import json,subprocess,sys
from export_contact_columns import change
from material_context import sha
base,out,source=map(lambda x:Path(x).resolve(),sys.argv[1:])
ids=json.loads((base/'identities.json').read_text())
if any(sha(base/k)!=v for k,v in ids.items()):raise ValueError('changed recruitment')
subprocess.run(['cp','-cR',base,out],check=True)
f=out/'numilab_human_myosim_visual_probe.mm';anchor='    if (useFEMSkinSupport || useFoundationLoad) {'
f.write_text(change(f.read_text(),anchor,'''    if (useFoundationLoad) {
        std::vector<std::pair<double,size_t>> ranked;
        for(size_t i=0;i<compiledActivation.generalizedAccelerationResidual.size();++i)
            ranked.emplace_back(std::abs(compiledActivation.generalizedAccelerationResidual[i]),i);
        std::sort(ranked.rbegin(),ranked.rend());
        for(size_t rank=0;rank<std::min<size_t>(8,ranked.size());++rank) {
            const auto i=ranked[rank].second;
            std::cout<<std::setprecision(12)<<"loaded_opening_coordinate="<<i
                     <<" name="<<(i<model.dofNames.size()?model.dofNames[i]:"unnamed")
                     <<" joint="<<model.dofs.at(i).jointIndex<<" q="<<model.dofs.at(i).qIndex
                     <<" acceleration="<<compiledActivation.generalizedAccelerationResidual[i]
                     <<" force="<<compiledActivation.generalizedForceResidual.at(i)
                     <<" bed="<<openingLoad[i]<<std::endl;
        }
    }
'''+anchor))
cmds=[[v.replace(str(base)+'/',str(out)+'/') for v in c] for c in json.loads((base/'build-commands.json').read_text())]
cwd=Path(json.loads((source/'launch.json').read_text())['environment']['DYLD_LIBRARY_PATH']).parent
for c in cmds[2:]:subprocess.run(c,cwd=cwd,check=True)
(out/'build-commands.json').write_text(json.dumps(cmds,indent=2)+'\n')
(out/'identities.json').write_text(json.dumps({k:sha(out/k) for k in ids},indent=2)+'\n')
