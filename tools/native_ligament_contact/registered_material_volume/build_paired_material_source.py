"""Route the verified coarse material parent, full local frame and face ancestry into the existing whole-volume constructor."""
from pathlib import Path
import json,sys,shutil,subprocess
import numpy as np
from emit_skin_source_registration import sha
src,reg,out,app=map(Path,sys.argv[1:]);out.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file() and f.suffix in ('.hpp','.metalinc','.metal') and not (out/f.name).exists():(out/f.name).hardlink_to(f)
(out/'metalrobo').mkdir(exist_ok=True)
for f in (src/'metalrobo').glob('*.metallib'):
    if not (out/'metalrobo'/f.name).exists():(out/'metalrobo'/f.name).hardlink_to(f)
proof=json.loads((reg/'native/physical-parent-identity.json').read_text())
for f,h in proof['inputs'].items():
    if sha(Path(f))!=h:raise ValueError('Material parent input changed '+f)
s=np.load(reg/'paired-skin-fat-refit-001/paired-geometry-and-mass.npz');g=np.load(reg/'paired-registration-with-face-ancestry/registration.npz')
q=json.loads((reg/'paired-skin-fat-refit-001/receipt.json').read_text())
assert q['native_F32_inverted_cells']==0 and q['native_F32_child_quality_minimum']>=1e-5
def n(x):return format(float(x),'.17g')
def table(name,typename,values,fmt):
    return 'inline constexpr std::array<'+typename+','+str(len(values))+'> '+name+'={{\n'+',\n'.join('    '+fmt(r) for r in values)+'\n}};\n'
def vec(row):return '{'+','.join(n(x) for x in row)+'}'
items=['#pragma once\nnamespace selected_material_parent {\nusing namespace ::metalrobo::human::coarse_tissue;\n']
for j,token in enumerate(('skin','fat')):
    for k,v in [('world',s['points_world_m'][j+1]),('local',g['physical_layers_donor_local_m'][j+1])]:items.append(table(token+'_'+k,'std::array<double,3>',v,vec))
    items.append(table(token+'_thickness','double',s['thickness_m'][j],n))
    items.append('inline constexpr double '+token+'_volume='+n(q[token+'_mass_kg']/(1000. if j==0 else 900.))+';\n')
items.append(table('face_parent','std::uint32_t',g['current_surface_face_parent'],lambda x:str(int(x))+'u'))
items.append('''inline constexpr auto child_source=[] {
    auto result=kElbowRefinementChildSourceFaceIdentities;
    for(std::size_t i=0;i<result.size();++i) {
        const auto row=kElbowRefinementChildParentSurfaceFaces[i];
        if(row>=face_parent.size()||kWholeBodySkinSourceFaces[row]!=result[i])
            throw "source cell has inconsistent original registered face ancestry";
        result[i]=face_parent[row];
    }
    return result;
}();
''')
items.append('inline constexpr std::string_view registration_sha256="'+proof['inputs'][str(reg/'paired-registration-with-face-ancestry/registration.npz')]+'";\n}\n')
h=out/'NumiHumanSelectedMaterialParent.hpp';h.write_text(''.join(items))
name='numilab_human_myosim_visual_probe.mm';code=(src/name).read_text()
anchor='#include "NumiHumanSelectedSkinReference.hpp"'
assert code.count(anchor)==1
code=code.replace(anchor,anchor+'\n#include "NumiHumanSelectedMaterialParent.hpp"')
at=code.index('        const auto skinSeedFace=');end=code.index('        const auto digest=loadedKneeSHA256(identityText.data()',at)
part=code[at:end]
pairs={'kWholeBodySkinCandidateSourcePayloadSHA256':'selected_skin_reference::payload','kWholeBodySkinDonorFrames':'selected_skin_reference::frames','kWholeBodySkinCandidateReferenceWorldPoints':'selected_skin_reference::points','kWholeBodySkinSourceEmbeddingResidualWorldM':'selected_skin_reference::residual','kWholeBodySkinInnerWorld':'selected_material_parent::skin_world','kWholeBodySkinInnerLocal':'selected_material_parent::skin_local','kWholeBodyFatDeepWorld':'selected_material_parent::fat_world','kWholeBodyFatDeepLocal':'selected_material_parent::fat_local','kWholeBodySkinThicknessM':'selected_material_parent::skin_thickness','kWholeBodyFatThicknessM':'selected_material_parent::fat_thickness','kWholeBodySkinLayerVolumeM3':'selected_material_parent::skin_volume','kWholeBodyFatLayerVolumeM3':'selected_material_parent::fat_volume','kWholeBodySkinSourceFaces':'selected_material_parent::face_parent'}
alias='''        const bool selectedParent=enableWholeBodySkinFat&&skinSourcePayloadSHA256==selected_skin_reference::payload;
        if(selectedParent)require(registeredVolumeInstallRequested&&preparedRegisteredVolumeBundle.has_value(),
            "selected skin requires its fully preflighted physical volume");
'''
import re
for i,(original,current) in enumerate(pairs.items()):
    local='active'+original[1:]
    if not re.search(r'\b'+original+r'\b',part):raise ValueError(original)
    part=re.sub(r'\b'+original+r'\b',local,part)
    alias+='        const auto& '+local+'=selectedParent?'+current+':'+original+';\n'
key='        if(enableWholeBodySkinFat) {\n            wholeBodySkinDetailUsed.assign'
assert part.count(key)==1
part=part.replace(key,'''        if(selectedParent) {
            region.sourceFaceIndices.assign(selected_material_parent::face_parent.begin(),selected_material_parent::face_parent.end());
        }
'''+key)
code=code[:at]+alias+part+'''        if(selectedParent)identityText.append(selected_material_parent::registration_sha256);
'''+code[end:];child='std::span<const std::uint32_t>(kElbowRefinementChildSourceFaceIdentities)}'
if code.count(child)!=1:raise ValueError('Unrecognized skin/fat cell ancestry')
code=code.replace(child,'std::span<const std::uint32_t>(selectedParent?selected_material_parent::child_source:kElbowRefinementChildSourceFaceIdentities)}')
(out/name).write_text(code)
cmd=[[a.replace(str(src)+'/',str(out)+'/') for a in row] for row in json.loads((src/'build-commands.json').read_text())]
(out/'build-commands.json').write_text(json.dumps(cmd,indent=2)+'\n')
for row in cmd:subprocess.run(row,cwd=app/'build-native',check=True)
(out/'identity.json').write_text(json.dumps({str(p):sha(p) for p in [src/name,out/name,h,out/'numi-human-native',*(out/'metalrobo').glob('*.metallib')]},indent=2)+'\n')
