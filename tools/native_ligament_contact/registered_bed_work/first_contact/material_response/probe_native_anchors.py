"""Observe the momentum change of native registered fixed anchors during the accepted root."""
import json
import os
from pathlib import Path
import subprocess
import sys
from register_material_surfaces import sha
from run_cavity_motion import execute_native


def probe(build,parent,out,measure_release=False):
    receipt=json.loads((build/'build.json').read_bytes())
    j=json.loads((parent/'launch.json').read_bytes())
    if sha(Path(j['argv'][0]))!=receipt['executable']['sha256']:raise ValueError('Unmatched native executable')
    out.mkdir();stage=out/'owner';stage.mkdir()
    src='numilab_human_myosim_visual_probe.mm'
    for p in build.iterdir():
        if p.suffix in {'.h','.hpp','.mm','.inc','.metalinc','.metal'} and p.name!=src:os.link(p,stage/p.name)
    text=(build/src).read_text();marker='        const char* endpointCapture=std::getenv("NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE");'
    if text.count(marker)!=1:raise ValueError('Ambiguous accepted state boundary')
    if measure_release:
        pose='require(status.succeeded(),status.message);return visualBodyStates(rigid.model,result.bodyPoses);'
        if text.count(pose)!=1:raise ValueError('Ambiguous native initial pose')
        observer=Path(__file__).with_name('audit_release_targets.inc').read_text()
        text=text.replace(pose,'require(status.succeeded(),status.message);\n'+observer+'\nreturn visualBodyStates(rigid.model,result.bodyPoses);')
    (stage/src).write_text(text.replace(marker,Path(__file__).with_name('audit_anchor_momentum.inc').read_text()+marker))
    commands=[]
    for original in receipt['commands']:
        cmd=[s.replace(str(build)+'/',str(stage)+'/') if s.startswith(str(build)+'/') else s for s in original]
        subprocess.run(cmd,check=True);commands.append(cmd)
    argv=[v.replace(str(parent)+'/',str(out)+'/') for v in j['argv']]
    argv[0]=str(stage/'numi-human-native')
    runtime={k:v.replace(str(parent)+'/',str(out)+'/') for k,v in j['environment'].items()}
    runtime.pop('NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS')
    runtime['NUMI_HUMAN_FIRST_ROOT_ANCHOR_MOMENTUM']='1'
    if measure_release:runtime['NUMI_HUMAN_FIRST_ROOT_RELEASE_TARGETS']='1'
    config=dict(j,argv=argv,environment=runtime,parent_launch=sha(parent/'launch.json'),
                compiled_observer=sha(Path(argv[0])),observer_source=sha(Path(__file__).with_name('audit_anchor_momentum.inc')),
                observer_build=commands)
    if measure_release:config['release_observer_source']=sha(Path(__file__).with_name('audit_release_targets.inc'))
    (out/'launch.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'native-traces').mkdir()
    env={k:v for k,v in os.environ.items() if not k.startswith(('NUMI_','NM_'))}
    result=execute_native(argv,j['cwd'],env|runtime,out,minimum_free_bytes=6*2**30,timeout_s=500)
    if result['exit']:return result['exit']
    a=out/'physical-accepted.node'
    if sha(a)!=sha(parent/a.name):raise ValueError('Observed endpoint differs from the original run')
    rows=[json.loads(line.split('=',1)[1]) for line in (out/'stdout.log').read_text().splitlines() if line.startswith('first_root_anchor_momentum=')]
    if not rows:raise ValueError('No registered fixed nodes captured')
    result.update(matched_native_endpoint=sha(a),objects=rows)
    (out/'anchor-momentum.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0


if __name__=='__main__':
    import argparse
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('build',type=Path)
    cli.add_argument('parent',type=Path)
    cli.add_argument('out',type=Path)
    cli.add_argument('--release-targets',action='store_true')
    args=cli.parse_args()
    sys.exit(probe(args.build.resolve(),args.parent.resolve(),args.out.resolve(),args.release_targets))
