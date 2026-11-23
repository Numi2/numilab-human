"""Transfer the existing coarse layer offsets and mechanical deltas to the admitted donor frames."""
import json,sys
from pathlib import Path
import numpy as np
from rebind_coarse_skin import floats,block
from compile_connected_joint import digest


def run(author,coarse,out):
    key='NumiHumanSelectedSkinReference.hpp';mat='NumiHumanSelectedMaterialParent.hpp';app='numilab_human_myosim_visual_probe.mm'
    old=(author/key).read_text();new=(coarse/key).read_text();layers=(author/mat).read_text();code=(author/app).read_text()
    a,b=[floats(block(s,'frames')).reshape(-1,19) for s in [old,new]];X=floats(block(old,'points')).reshape(-1,3)
    if not np.array_equal(X,floats(block(new,'points')).reshape(-1,3)) or not np.array_equal(a[:,0],b[:,0]):raise ValueError('World reference or material owners changed')
    def local(f,delta):return np.einsum('rij,rj->ri',f[:,4:13].reshape(-1,3,3),delta)
    def numbers(x):return '{'+','.join(format(float(v),'.17g') for v in x)+'}'
    def data(rows):return '\n    '+',\n    '.join(map(numbers,rows))+'\n'
    def replace(s,name,value):
        start=s.index(' '+name+'={{')+len(' '+name+'={{');end=s.index('}};',start);return s[:start]+data(value)+s[end:]
    reduction=(author/'NumiHumanWholeBodySkinReduction.hpp').read_text()
    dw,dl=[floats(block(reduction,'kWholeBodySkinMechanical'+k+'Deltas')).reshape(-1,3) for k in ('World','DonorLocal')]
    if abs(local(a,dw)-dl).max()>2e-7:raise ValueError('Original mechanical delta registration failed')
    dlnew=local(b,dw);payload={'mechanical_world':dw,'mechanical_local':dlnew,'donor':b[:,0].astype('u4'),'world_to_donor':b[:,4:13]};errors={}
    for tag in ('skin','fat'):
        w,l=[floats(block(layers,tag+'_'+k)).reshape(-1,3) for k in ['world','local']];err=np.linalg.norm(a[:,1:4]+local(a,w-X)-l,axis=1)
        if max(err)>2e-7:raise ValueError('Unmatched original material registration')
        ln=b[:,1:4]+local(b,w-X);layers=replace(layers,tag+'_local',ln);payload.update({tag+'_world':w,tag+'_local':ln});errors[tag]=float(max(err))
    out.mkdir();np.savez_compressed(out/'registered-parent.npz',**payload)
    import re
    layers=re.sub(r'(registration_sha256=")[0-9a-f]{64}',r'\g<1>'+digest(out/'registered-parent.npz'),layers)
    header=new[:new.rindex('}')]+f'inline constexpr std::array<std::array<double,3>,{len(b)}> mechanical_local={{{{'+data(dlnew)+'}};\n}\n'
    (out/key).write_text(header);(out/mat).write_text(layers)
    before='kWholeBodySkinMechanicalWorldDeltas,kWholeBodySkinMechanicalDonorLocalDeltas,'
    if code.count(before)!=2:raise ValueError('Unknown mechanical delta consumers')
    for condition in ('rebound','selectedParent'):
        code=code.replace(before,'kWholeBodySkinMechanicalWorldDeltas,('+condition+'?selected_skin_reference::mechanical_local:kWholeBodySkinMechanicalDonorLocalDeltas),',1)
    (out/app).write_text(code)
    cfg=json.loads((coarse/'result.json').read_bytes());cfg.update(original_parent=digest(author/mat),parent=digest(out/mat),source_coarse_registration=digest(coarse/'result.json'),parent_archive=digest(out/'registered-parent.npz'),header=digest(out/key),application_from=digest(author/app),application=digest(out/app),control_local_errors_m=errors)
    (out/'result.json').write_text(json.dumps(cfg,indent=2)+'\n')
if __name__=='__main__':run(*(Path(k).resolve() for k in sys.argv[1:]))
