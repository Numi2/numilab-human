"""Prepare a matched native full-volume trial with the qualified starting tissue and skeletal pose."""
import json,subprocess,sys
from pathlib import Path
from evaluate_registered_bed_transfer import digest
from screen_registered_dense_posture import initial_body_targets

def prepare(full,body,pose,owner,out):
    out.mkdir(exist_ok=False);launch=json.loads((full/'launch.json').read_text())
    r=json.loads((pose/'recovery.json').read_text());b=json.loads(((body/'launch.json') if body.is_dir() else body).read_text())
    rec=json.loads((pose/'loaded-registration.json').read_text())
    _,identity=initial_body_targets(body)
    if any(r['source'].get(k)!=v for k,v in identity.items()) or rec['outer_rows_below_plane'] or rec['largest_attachment_error_m']>2e-7:raise ValueError('unmatched supported posture')
    payload=pose/'current.node';compressed=pose/r['asset']
    if not payload.exists():
        if not compressed.exists():subprocess.run(['/opt/homebrew/bin/gh','release','download',r['tag'],'-R',r['repository'],'-p',r['asset'],'-D',str(pose)],check=True)
        if digest(compressed)!=r['compressed_sha256']:raise ValueError('compressed input drift')
        subprocess.run(['/opt/homebrew/bin/zstd','-qd',str(compressed),'-o',str(payload)],check=True)
    if digest(payload)!=r['decoded_node_sha256']:raise ValueError('loaded node identity')
    argv=launch['argv']
    if body.is_dir():
        while '--pose-q' in argv:
            i=argv.index('--pose-q');del argv[i:i+3]
        src=b['argv'][b['argv'].index('--resting-initial-posture')+1]
        argv+=['--resting-initial-posture',src]
        launch['pins'][src]=digest(Path(src))
        launch['environment'].pop('NUMI_EXPERIMENT_INITIAL_ROOT_POSE',None)
    else:
        launch['environment']['NUMI_EXPERIMENT_INITIAL_ROOT_POSE']=b['environment']['NUMI_EXPERIMENT_INITIAL_ROOT_POSE']
    idx=argv.index('--root-pose');start=b['argv'].index('--root-pose')
    argv[idx+1:idx+8]=b['argv'][start+1:start+8]
    argv[argv.index('--muscle-step-count')+1]='1'
    launch['environment']['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']=str(payload)
    launch['environment']['NUMI_HUMAN_REGISTERED_CELL_QUALITY']='1'
    launch['environment']['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']='1'
    launch['pins'][str(payload)]=r['decoded_node_sha256']
    launch['current_pose_sha256']=r['decoded_node_sha256']
    launch['full_scene_receipt']=digest(full/'launch.json')
    launch['target_pose_receipt']=identity['target_receipt_sha256']
    (out/'config.json').write_text(json.dumps(launch,indent=2)+'\n')
if __name__=='__main__':prepare(*(Path(v).resolve() for v in sys.argv[1:]))
