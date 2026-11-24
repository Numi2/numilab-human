"""Apply the accepted geometry publication ordering fix to a registered native cohort."""
import argparse
import json
import subprocess
from pathlib import Path
from compile_connected_joint import digest

HEADER = 'NumiHumanRestingVisual.hpp'
HOST = 'numilab_human_myosim_visual_probe.mm'


def correct_order(source):
    start='        if(captureThisFrame) {\n            NSDictionary* surfaceAuditOutcome=@{'
    wall='        require(wallCorrection.w==0,"accepted ventricular material volume closure failed");\n'
    skin='        require(skinAudit.w==0&&skinAudit.z==0,"accepted full skin intersects the bed beyond the 1 mm inspection tolerance");\n'
    if source.count(start)!=1 or source.count(wall+skin)!=1 or source.index(start)>source.index(wall):
        raise ValueError('Unexpected native surface certification sequence')
    return source.replace(wall+skin,'').replace(start,wall+skin+start)


def build(parent,dest,control):
    dest.mkdir()
    for f in parent.iterdir():
        if f.name in (HEADER,HOST,'metalrobo','numi-human-native') or f.suffix in ('.o','.json','.log'):
            continue
        if f.is_file():
            (dest/f.name).symlink_to(f.resolve())
    (dest/'metalrobo').symlink_to((parent/'metalrobo').resolve(),target_is_directory=True)
    (dest/HOST).write_bytes((parent/HOST).read_bytes())
    (dest/HEADER).write_text(correct_order((parent/HEADER).read_text()))
    cfg=json.loads((parent/'registered-sections.json').read_bytes())
    commands=[[a.replace(str(parent)+'/',str(dest)+'/') for a in cmd] for cmd in cfg['commands']]
    cwd=Path(json.loads((control/'launch.json').read_bytes())['environment']['DYLD_LIBRARY_PATH']).parent
    with (dest/'build.log').open('w')as log:
        for cmd in commands:
            subprocess.run(cmd,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True)
    (dest/'verification-build.json').write_text(json.dumps(dict(source=digest(parent/HEADER),header=digest(dest/HEADER),
        control_binary=digest(parent/'numi-human-native'),binary=digest(dest/'numi-human-native'),
        gpu=digest(dest/'metalrobo/MetalRobo.metallib'),commands=commands),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','dest','control'):
        p.add_argument(name,type=lambda s:Path(s).resolve())
    build(**vars(p.parse_args()))
