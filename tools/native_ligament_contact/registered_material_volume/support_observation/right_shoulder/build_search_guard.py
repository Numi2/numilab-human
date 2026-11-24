"""Protect the search direction against outward sub-ulp clearance without changing material acceptance."""
import json,shutil,subprocess,sys
from pathlib import Path
from compile_connected_joint import digest
from build_extremum_direction import modify


def build(parent,out):
    cfg=json.loads((parent/'build.json').read_bytes())
    orig=parent/'src/metal/fem.metalinc'
    if digest(parent/'NumiMatter.metallib')!=cfg['pins']['NumiMatter.metallib']:raise ValueError('Changed native library')
    recovery=next((parent.parent/'publication/evidence').glob('*/*/recovery.json'))
    generated=json.loads(recovery.read_bytes())['verified_member_sha256']['build/src/metal/numi_human.metalinc']
    if digest(parent/'src/metal/numi_human.metalinc')!=generated:raise ValueError('Changed human force operator')
    out.mkdir(exist_ok=True);src=out/'src';(src/'metal').mkdir(parents=True,exist_ok=True)
    for p in (parent/'src').iterdir():
        if p.name=='matter_kernels.metal':shutil.copy2(p,src/p.name)
        elif p.name!='metal' and not (src/p.name).exists():(src/p.name).symlink_to(p.resolve())
    for p in orig.parent.iterdir():
        dst=src/'metal'/p.name
        if p.name!=orig.name and not dst.exists():dst.symlink_to(p.resolve())
    text,_=modify(orig.read_text())
    marker='    if(mode!=2u || !(original.value>lo && original.value<hi))return original;'
    if text.count(marker)!=1:raise ValueError('Unmatched material search operator')
    guard=Path(__file__).with_name('search_endpoint.metalinc').read_text()
    text=text.replace(marker,'''    if (!(original.value>lo && original.value<hi)) return original;
    const NMRegisteredVolumeControl endpoint=registeredEndpointControl(start,end,lo,hi,original);
    if (!(endpoint.value>lo && endpoint.value<hi) || mode!=2u) return endpoint;''')
    key='inline NMRegisteredVolumeControl registeredTrialControl('
    text=text.replace(key,guard+'\n'+key)
    # The fallback geometric endpoint has time coordinate one.
    text=text.replace('bound.z)):-INFINITY,3u};','bound.z)):-INFINITY,3u,1.f};')
    file=src/'metal/fem.metalinc';file.write_text(text)
    commands=[[s.replace(str(parent)+'/',str(out)+'/')for s in cmd]for cmd in cfg['commands'][:2]]
    for i,cmd in enumerate(commands):
        with (out/f'build-{i}.log').open('w')as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'build.json').write_text(json.dumps(dict(parent=digest(parent/'build.json'),commands=commands,source=digest(file),library=digest(out/'NumiMatter.metallib')),indent=2)+'\n')
if __name__=='__main__':build(*(Path(a).resolve()for a in sys.argv[1:]))
