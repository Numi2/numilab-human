"""Compile the registered native app with the corrected starting-material asset pin."""
from pathlib import Path
import sys,json,os,subprocess
from compile_connected_joint import digest

def run(base,pose,out,contact=None):
    s=base/'app';out.mkdir();app=out/'app';app.mkdir()
    key='NumiHumanLoadedSkinPoseData.inc';src=s/key
    for f in s.iterdir():
        if f.name!=key and (contact is None or f.name!='NumiHumanCorticalSkinBinding.inc') and f.suffix in ('.h','.hpp','.inc','.mm'):os.link(f.resolve(),app/f.name)
    if contact:
        from build_cortical_skin_registration import write_contact_include
        write_contact_include(contact,app)
    ref=json.loads((base/'scene/launch.json').read_bytes())['environment']['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']
    old='char nodeSHA[]="'+digest(Path(ref))+'"';new='char nodeSHA[]="'+digest(pose)+'"';txt=src.read_text()
    if txt.count(old)!=1:raise ValueError('Incorrect original material pin')
    txt=txt.replace(old,new)
    if contact:
        contact_data=json.loads((contact/'result.json').read_bytes());env=json.loads((base/'scene/launch.json').read_bytes())['environment']
        reference='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE_SHA256'
        old_ref='char referenceSHA[]="'+env[reference]+'"'
        if txt.count(old_ref)!=1 or contact_data['loaded']!=digest(pose) or contact_data['elements']!=env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_ELEMENTS_SHA256']:raise ValueError('Mismatched assembled material pair')
        txt=txt.replace(old_ref,'char referenceSHA[]="'+contact_data['material_reference']+'"')
    (app/key).write_text(txt);cmds=[[w.replace(str(s)+'/',str(app)+'/') for w in c] for c in json.loads((base/'build.json').read_bytes())['commands']]
    with (out/'build.log').open('w') as log:
        for c in cmds:subprocess.run(c,stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'build.json').write_text(json.dumps(dict(commands=cmds,input=digest(src),output=digest(app/key),binary=digest(app/'numi-human-native')),indent=2)+'\n')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
