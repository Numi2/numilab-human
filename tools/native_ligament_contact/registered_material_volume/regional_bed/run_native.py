"""Run one matched regional bed trial and retain the exact physical inputs."""
import json,os,sys
from pathlib import Path
from run_cavity_motion import execute_native
from compile_registered_boundary import sha

def run(parent,owner,material,out,steps,seating=None):
    cfg=json.loads((parent/'launch.json').read_bytes())
    a=cfg['argv'];e=cfg['environment']; pins=cfg['pins']
    if e['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1' or e['NUMI_HUMAN_LOCAL_BED_REGIONAL_THICKNESS']!='1':
        raise ValueError('Regional bed and variable pleural pressure are required')
    if out.exists(): raise ValueError('Output directory already exists')
    for name,expected in pins.items():
        if sha(Path(name))!=expected:raise ValueError('Stale source '+name)
    previous_output=Path(a[4]).resolve().parent
    a[0]=str(owner/'numi-human-native')
    a[4]=str(out/'native-traces')
    a[a.index('--muscle-step-count')+1]=str(steps)
    if '--persistent-metal-stand-steps' in a:a[a.index('--persistent-metal-stand-steps')+1]=str(steps)
    for k,v in list(e.items()):
        if v.startswith(str(previous_output)+'/') and v not in pins:
            e[k]=str(out/Path(v).relative_to(previous_output))
    reuse=os.environ.get('NUMI_HUMAN_RESTING_CAPTURE_REUSE_SOURCE_DIR')
    if reuse:
        e['NUMI_HUMAN_RESTING_CAPTURE_REUSE_SOURCE_DIR']=str(Path(reuse).resolve())
    e['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']=str(steps)
    e['NUMI_HUMAN_REGISTERED_BED_MATERIAL_RECEIPT']=str(material/'material-boundary.json')
    reg=json.loads((material/'material-boundary.json').read_bytes())
    for p in [owner/'numi-human-native',owner/'NumiHumanRestingVisual.hpp',
              owner/'numilab_human_myosim_visual_probe.mm',owner/'metalrobo/HumanRestingSkin.metallib',owner/'metalrobo/MetalRobo.metallib',material/'registered-boundary.nhskin',
              material/'material-boundary.json',Path(reg['contact_columns_path'])]:pins[str(p)]=sha(p)
    cfg.update(parent_launch=sha(parent/'launch.json'),registered_regional_material=reg['nhskin_sha256'])
    if seating is not None:
        import numpy as np
        from scipy.spatial.transform import Rotation
        solution=json.loads(seating.read_bytes())
        fit,release=solution['fit'],solution['release']
        witness_case=seating.parent/'trial-8'
        if release['launch']!=sha(witness_case/'launch.json') or fit['initial_witness_sha256']!=release['witness']:
            raise ValueError('Unmatched initial bed witness')
        prior=json.loads((witness_case/'launch.json').read_bytes())
        if prior['registered_regional_material']!=reg['nhskin_sha256'] or fit['max_stock_fraction']>=1:
            raise ValueError('Foreign material or inadmissible seating')
        root=json.loads((witness_case/'native-initial-bodies.json').read_bytes())['all_bodies'][0]['current']
        center=np.array(fit['center_world_m']);rot=Rotation.from_rotvec(fit['rotation_world_vector_rad'])
        xyz=center+rot.apply(np.array(root['position_world_m'])-center)-fit['translation_toward_plane_m']*np.array(release['normal'])
        quat=rot*Rotation.from_quat(root['orientation_world_xyzw'])
        i=a.index('--root-pose')+1
        a[i:i+7]=[str(v) for v in np.r_[xyz,quat.as_quat()]]
        cfg['material_seating_source']=sha(seating)
    out.mkdir();(out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    actual={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|e
    result=execute_native(a,cfg['cwd'],actual,out,minimum_free_bytes=4*2**30,timeout_s=300)
    (out/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)
    return result['exit']

if __name__=='__main__':sys.exit(run(*(Path(v).resolve() for v in sys.argv[1:5]),int(sys.argv[5]),Path(sys.argv[6]).resolve() if len(sys.argv)>6 else None))
