"""Bind the selected native source to a recovered fine material volume by its exact coarse parent."""
import hashlib,json,sys
from pathlib import Path
import numpy as np

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main(current,baseline,fine_state,output):
    state=json.loads(fine_state.read_text())
    pins=state['current_reference']['inputs_sha256']
    def packet(home,sub,name):return home/sub/name
    cell='paired-skin-fat-refit-001/paired-geometry-and-mass.npz'
    basis='paired-registration-with-face-ancestry/registration.npz'
    for name in (cell,basis):
        p=baseline/name
        matching=[h for s,h in pins.items() if s.endswith('/'+name)]
        if matching!=[sha(p)]:raise ValueError('Fine volume has a different parent '+name)
    identities=json.loads((current/'identity.json').read_text());selected=identities['current']
    if sha(Path(selected['path']))!=selected['sha256']:raise ValueError('Current NHSKIN changed')
    old,new=np.load(baseline/cell),np.load(current/cell)
    if set(old.files)!=set(new.files) or any(not np.array_equal(old[k],new[k]) for k in old.files):
        raise ValueError('Parent positions, topology or masses changed; regenerate fine volume')
    reg,prev=np.load(current/basis),np.load(baseline/basis)
    maxdiff={}
    for k in reg.files:
        if not np.isfinite(reg[k]).all():raise ValueError(k)
        if k in ('normal_world','donor_local_m','projected_residual_world_m','physical_layers_donor_local_m'):
            maxdiff[k]=float(np.abs(reg[k]-prev[k]).max())
        elif not np.array_equal(reg[k],prev[k]):raise ValueError('Unmatched registered physical parent '+k)
    local_change=np.linalg.norm(reg['physical_layers_donor_local_m']-prev['physical_layers_donor_local_m'],axis=-1)
    # The original fine ledger retains donor-local attachment coordinates. Quantify
    # their change against the existing native 20-micrometre registration gate.
    if local_change.max()>2e-5:raise ValueError('Re-register the material attachment positions')
    report=dict(source=selected,inputs={str(p):sha(p) for p in (current/cell,baseline/cell,current/basis,baseline/basis,fine_state)},matched_cell_arrays=sorted(old.files),maximum_absolute_component_changes=maxdiff,
                maximum_donor_position_change_m=float(local_change.max()),volume_admitted=False)
    def numbers(a):return '{'+','.join(format(float(x),'.17g') for x in a)+'}'
    n=len(reg['source_face'])
    def array(name,kind,rows):
        return 'inline constexpr std::array<'+kind+','+str(n)+'> '+name+'={{\n  '+',\n  '.join(rows)+'\n}};\n'
    lines=['#pragma once\nnamespace selected_skin_reference {\n','using namespace ::metalrobo::human::coarse_tissue;\n','inline constexpr std::string_view payload="'+selected['sha256']+'";\n']
    lines.append(array('frames','WholeBodySkinDonorFrameV1',('{'+str(int(b))+'u,'+','.join(numbers(np.asarray(x,dtype='f4')) for x in (l,m,p,v))+'}' for b,l,m,p,v in zip(reg['donor_body'],reg['donor_local_m'],reg['world_to_donor'],reg['world_m'],reg['normal_world']))))
    lines.append(array('points','std::array<double,3>',map(numbers,reg['world_m'])))
    lines.append(array('residual','std::array<double,3>',map(numbers,reg['projected_residual_world_m'])))
    lines.append('}\n');output.mkdir(exist_ok=True);hpp=output/'NumiHumanSelectedSkinReference.hpp'
    hpp.write_text(''.join(lines));report['header_sha256']=sha(hpp)
    (output/'physical-parent-identity.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
