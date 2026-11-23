"""Submit the new volume to the complete native scene and retain its first physical admission result."""
import sys,os,json
from pathlib import Path
from run_cavity_motion import execute_native
from compile_connected_joint import digest

def run(base,build,material,pose,out):
    cfg=json.loads((base/'launch.json').read_bytes());argv=cfg['argv'];env=cfg['environment'];pins=cfg['pins']
    reg=json.loads((build/'registration.json').read_bytes());qual=json.loads((material/'result.json').read_bytes())
    if reg['skin_pose']!=digest(pose/'registration.json') or qual['exit'] or digest(Path(argv[1]))!=reg['physical_source']:raise ValueError('Unmatched scene components')
    out.mkdir()
    argv[0]=str(build/'numi-human-native');argv[4]=str(out/'native-traces');pins[argv[0]]=reg['executable']
    pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    for k,name in [('TETGEN_NODE','mesh.node'),('TETGEN_ELEMENTS','mesh.ele'),('CHILD_LABELS','current-child.labels.csv')]:
        path=str(material/name);expected=qual['output_hashes'][name]
        if digest(Path(path))!=expected:raise ValueError(f'Unmatched volume {k}')
        env[pre+k]=path;env[pre+k+'_SHA256']=pins[path]=expected
    env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']=str(material/'mesh.node')
    cfg['environment']={k:v.replace(str(base)+'/',str(out)+'/') for k,v in env.items()}
    cfg['joined_pose_registration']=digest(pose/'registration.json');cfg['material_receipt']=digest(material/'result.json')
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    result=execute_native(argv,cfg['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|cfg['environment'],out,minimum_free_bytes=4*2**30,timeout_s=300);print(result)
if __name__=='__main__':run(*map(lambda k:Path(k).resolve(),sys.argv[1:]))
