"""Load a freshly verified registered volume into the intact native Human scene."""
import json
import os
import sys
from pathlib import Path
from bind_registered_shoulder import sha
from run_cavity_motion import execute_native


def prepare(control, certificate, output, support=None, carriers=None, witnesses=None, mass=None, release=None):
    config=json.loads((control/'launch.json').read_bytes())
    check=json.loads(certificate.read_bytes())
    settings=check['settings']
    for key,value in settings.items():
        if not key.endswith('_SHA256'):
            path=Path(value)
            if not path.is_file():raise FileNotFoundError(path)
    runtime = dict(config['environment'])
    for key,value in runtime.items():
        runtime[key]=value.replace(str(control)+'/',str(output)+'/')
    runtime.update(settings)
    extra_pins={}
    if support is not None:
        material=json.loads((support/'receipt.json').read_bytes())
        for name in ('skin','fat'):
            tag=name.upper()
            role=material[name]
            for path_key,hash_key in [('p1_one_hop_path','p1_one_hop_sha256'),('role_geometry_path','role_geometry_sha256'),('state_path','state_sha256')]:
                path=Path(role[path_key])
                if sha(path)!=role[hash_key]:raise ValueError('Changed '+path_key)
                extra_pins[str(path)]=role[hash_key]
            prefix='NUMI_H706_FRACTIONAL_'
            runtime[prefix+'P1_'+tag+'_NPZ']=role['p1_one_hop_path']
            runtime[prefix+'P1_'+tag+'_SHA256']=role['p1_one_hop_sha256']
            runtime[prefix+'STATE_SHA256']=role['state_sha256']
            runtime[prefix+'VOLUME_'+tag+'_SHA256']=role['role_geometry_sha256']
            runtime[prefix+'GEOMETRY_'+tag+'_SHA256']=role['role_geometry_sha256']
    if carriers is not None:
        if support is None:raise ValueError('Rebound carriers require their current P1 geometry')
        for tag,r in json.loads((carriers/'carriers.json').read_bytes()).items():
            path=Path(r['path'])
            if sha(path)!=r['sha256'] or r['missing_carriers']:raise ValueError('Unqualified carrier primary')
            runtime['NUMI_H706_FRACTIONAL_CARRIERS_'+tag+'_NPZ']=str(path)
            runtime['NUMI_H706_FRACTIONAL_CARRIERS_'+tag+'_SHA256']=r['sha256']
            extra_pins[str(path)]=r['sha256']
    if witnesses is not None:
        if carriers is None:raise ValueError('Witnesses require their pinned carrier packet')
        for tag,info in json.loads((witnesses/'witnesses.json').read_bytes()).items():
            path=Path(info['path'])
            if sha(path)!=info['sha256']:raise ValueError('Stale carrier witness table')
            runtime['NUMI_H706_FRACTIONAL_CARRIER_WITNESS_'+tag+'_CSV']=str(path)
            runtime['NUMI_H706_FRACTIONAL_CARRIER_WITNESS_'+tag+'_SHA256']=info['sha256']
            extra_pins[str(path)]=info['sha256']
    if mass is not None:
        verdict=json.loads((mass/'result.json').read_bytes())
        if verdict['exit'] or len(verdict['receipts'])!=2:raise ValueError('Native mass replay is incomplete')
        fields=('CHILD_MASS','TRANSPORT_FLOW','SOURCE_CARRIER_FLOW','CHILD_SOURCE_DONOR','SOURCE_PARENT','SOURCE_PARENT_DONOR')
        for name,digest in verdict['receipts'].items():
            tag=name.split('-')[0].upper()
            prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_RECORDED_ALLOCATION_'+tag+'_'
            certified=mass/'role-flows'/name
            if sha(certified)!=digest:raise ValueError('Changed native mass receipt')
            role=json.loads(certified.read_bytes())
            runtime[prefix+'RECEIPT_JSON']=str(certified);runtime[prefix+'RECEIPT_SHA256']=digest
            extra_pins[str(certified)]=digest
            for field in fields:
                f=mass/'role-flows'/(tag.lower()+'-'+field.lower().replace('_','-')+'.csv')
                digest=role['flow_csv'][f.name]['sha256']
                if sha(f)!=digest:raise ValueError('Changed production role mass input')
                runtime[prefix+field+'_CSV']=str(f);runtime[prefix+field+'_CSV_SHA256']=digest
                extra_pins[str(f)]=digest
    args=config['argv'].copy()
    if release is not None:
        if mass is None:raise ValueError('Physical release requires the certified mass')
        build=json.loads((release/'build.json').read_bytes())
        for key in ('executable','release_pose','library'):
            f=Path(build[key]['path'])
            if sha(f)!=build[key]['sha256']:raise ValueError('Changed release owner')
            extra_pins[str(f)]=build[key]['sha256']
        args[0]=build['executable']['path']
        selected_library_dir=str(Path(build['library']['path']).parent)
        old_library_path=runtime.get('DYLD_LIBRARY_PATH','').split(os.pathsep)
        runtime['DYLD_LIBRARY_PATH']=os.pathsep.join(dict.fromkeys([selected_library_dir,*filter(None, old_library_path)]))
        runtime['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']=build['release_pose']['path']

    args[4]=str(output/'native-traces')
    args[args.index('--muscle-step-count')+1]='1'
    runtime['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']='1'
    # The input already contains the cavity repairs certified by the importer.
    # An overlay from the earlier connectivity would target the wrong cell rows.
    for key in ('NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ','NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256',
                'NUMI_HUMAN_SKIN_FAT_FIXED_CAVITY_MASS_MOMENTS'):
        runtime.pop(key,None)
    pins=dict(config['pins'],**extra_pins)
    for key,value in settings.items():
        if not key.endswith('_SHA256'):
            pins[value]=sha(Path(value))
    check_control=sha(control/'launch.json')
    for path,digest in config['pins'].items():
        if sha(Path(path))!=digest:raise ValueError(f'Parent input changed {path}')
    launch=dict(config,argv=args,environment=runtime,pins=pins,parent_launch=check_control,registered_volume_verification=sha(certificate))
    output.mkdir()
    (output/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    env={k:v for k,v in os.environ.items() if not k.startswith(('NUMI_','NM_'))}
    result=execute_native(args,launch['cwd'],env|runtime,output,minimum_free_bytes=8*2**30,timeout_s=500)
    print(result,flush=True)
    return result['exit']


if __name__=='__main__':sys.exit(prepare(*(Path(p).resolve() for p in sys.argv[1:])))
