"""Build the paired native material registration without mutating previous binaries or their headers."""
import json,os,subprocess,sys
from pathlib import Path
from compile_connected_joint import digest
def build(parent,coarse,out,ligaments=None):
    key='NumiHumanSelectedSkinReference.hpp';cfg=json.loads((parent/'registration.json').read_bytes());receipts=json.loads((coarse/'result.json').read_bytes())
    if digest(parent/key)!=receipts['original'] or digest(coarse/key)!=receipts['header'] or digest(parent/'numi-human-native')!=cfg['executable']:raise ValueError('Unmatched native registration')
    replacements={key:coarse/key}
    for name,old,new in [('NumiHumanSelectedMaterialParent.hpp','original_parent','parent'),('numilab_human_myosim_visual_probe.mm','application_from','application')]:
        if new in receipts:
            if digest(parent/name)!=receipts[old] or digest(coarse/name)!=receipts[new]:raise ValueError('Unmatched physical layer source')
            replacements[name]=coarse/name
    if ligaments:
        lig=json.loads((ligaments/'registration.json').read_bytes())
        key_lig='NumiHumanRegisteredLigaments.hpp';mt='NumiHumanRegisteredLigamentMaterial.inc'
        if digest(parent/key_lig)!=lig['source_header'] or digest(ligaments/key_lig)!=lig['header'] or digest(ligaments/mt)!=lig['material']:raise ValueError('Unmatched ligament definition')
        cfg['ligament_registration']=digest(ligaments/'registration.json')
        replacements[key_lig]=ligaments/key_lig
    out.mkdir()
    if ligaments:os.link(ligaments/mt,out/mt)
    for f in parent.iterdir():
        if f.suffix in ('.hpp','.inc','.mm','.h'):os.link(replacements.get(f.name,f),out/f.name)
    cmds=[]
    for cmd in cfg['commands']:
        cmd=[str(out/Path(v).name) if Path(v).parent==parent and Path(v).name in ('numilab_human_myosim_visual_probe.mm','numi-human-native.o','numi-human-native') else v for v in cmd]
        with (out/'compiler.log').open('a') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        cmds.append(cmd)
    cfg.update(commands=cmds,coarse_registration=digest(coarse/'result.json'),parent_executable=cfg['executable'],executable=digest(out/'numi-human-native'))
    (out/'registration.json').write_text(json.dumps(cfg,indent=2)+'\n')
if __name__=='__main__':build(*(Path(k).resolve() for k in sys.argv[1:]))
