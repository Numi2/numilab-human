"""Capture the actual Metal FK before the skin registration check, using the normal native scene."""
from pathlib import Path
import json,os,sys,subprocess
from compile_connected_joint import digest
from prepare_live_cortex import exact
from run_cavity_motion import execute_native
def run(build,base,out):
    out.mkdir();app=out/'app';app.mkdir();src=build/'app';unit='numilab_human_myosim_visual_probe.mm'
    for f in src.iterdir():
        if f.name!=unit and f.suffix in ('.hpp','.inc','.mm'):(app/f.name).symlink_to(f.resolve())
    s=(src/unit).read_text()
    prefix='    numiLoadedSkin::sourceFrames(bodies);'
    s=exact(s,prefix,'''    if(const char* witness=std::getenv("NUMI_HUMAN_MATERIAL_TARGET_POSE_DUMP"))
        writeSkinPoseSnapshot(skin,bodies,restBodies,std::filesystem::path(witness));
'''+prefix);(app/unit).write_text(s)
    data=json.loads((build/'build.json').read_text());cs=[[a.replace(str(src)+'/',str(app)+'/') for a in c] for c in data['commands'][-2:]]
    with (out/'build.log').open('wb') as log:
        for c in cs:subprocess.run(c,stdout=log,stderr=subprocess.STDOUT,check=True)
    cfg=json.loads((base/'launch.json').read_text());job=out/'scene';job.mkdir();argv=cfg['argv'];argv[0]=str(app/'numi-human-native');argv[4]=str(job/'native-traces')
    env={k:v.replace(str(base)+'/',str(job)+'/') for k,v in cfg['environment'].items()}
    env['NUMI_HUMAN_MATERIAL_TARGET_POSE_DUMP']=str(job/'target-pose.json')
    cfg['environment']=env;cfg['pins'][argv[0]]=digest(Path(argv[0]));cfg['parent_launch']=digest(base/'launch.json')
    (job/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    execute_native(argv,cfg['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|env,job,minimum_free_bytes=4*2**30,timeout_s=120)
    if not (job/'target-pose.json').exists():raise ValueError('Native scene did not emit target FK')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
