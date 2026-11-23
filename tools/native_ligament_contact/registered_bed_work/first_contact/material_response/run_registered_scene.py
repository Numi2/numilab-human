"""Enter the existing native full-scene loader with a matched set of experimental shoulder assets."""
import json,os,sys
from pathlib import Path
from compile_connected_joint import digest
from run_cavity_motion import execute_native


def run(job,rigid,bones,vessels,out,attachments=None,tissues=None):
    base=json.loads((job/'launch.json').read_text())
    argv=base['argv'].copy()
    env={k:v.replace(str(job)+'/',str(out)+'/') for k,v in base['environment'].items()}
    new_bone=next(bones.glob('*.nhbones'))
    inputs={argv[1]:rigid/'source-bound.nhrigid',argv[3]:new_bone}
    anatomy=tissues or vessels
    if tissues:
        registered=json.loads((tissues/'result.json').read_bytes())
        if registered['rigid']!=digest(rigid/'source-bound.nhrigid') or registered['bones']!=digest(new_bone):
            raise ValueError('Tissue carrier and native skeleton differ')
        for flag in ('--skin-payload','--soft-tissue-payload'):
            original=argv[argv.index(flag)+1]
            inputs[original]=tissues/Path(original).name
    for flag,new in [('--torso-anatomy-payload',next(anatomy.glob('*.nhanatomy'))),('--resting-anatomy-receipt',anatomy/'resting-anatomy-receipt.json')]:
        inputs[argv[argv.index(flag)+1]]=new
    if attachments:
        tendon_path=next(attachments.glob('*.nhtendon'))
        mt=json.loads((attachments/'result.json').read_bytes())
        if mt['bones']!=digest(new_bone) or mt['rigid']!=digest(rigid/'source-bound.nhrigid'):
            raise ValueError('Mismatched muscle registration')
        inputs[argv[2]]=next(attachments.glob('*.nhmyo'))
        inputs[argv[argv.index('--tendon-payload')+1]]=tendon_path
    pins=base['pins'].copy()
    for old,new in inputs.items():
        if argv.count(old)!=1:raise ValueError('Ambiguous native input')
        if old in pins and digest(Path(old))!=pins[old]:raise ValueError('Changed control')
        argv[argv.index(old)]=str(new)
        pins.pop(old,None)
        pins[str(new)]=digest(new)
    check=json.loads((vessels/'result.json').read_bytes())
    if check['rigid_sha256']!=digest(rigid/'source-bound.nhrigid') or check['candidate_parent']!=digest(vessels/'parent-7.nhtissue'):
        raise ValueError('Mismatched vessel parent frame')
    pins[str(vessels/'parent-7.nhtissue')]=check['candidate_parent']
    if tissues:
        for entry in json.loads((tissues/'resting-anatomy-receipt.json').read_bytes())['passive_tissue_parent_bindings']['source_payloads']:
            if entry['sha256']!=digest(Path(entry['path'])):raise ValueError('Unmatched generated carrier')
            pins[entry['path']]=entry['sha256']
    argv[4]=str(out/'native-traces')
    # This is an admission trial. Use the same physical root size and budget one root.
    argv[argv.index('--muscle-step-count')+1]='1'
    out.mkdir();(out/'native-traces').mkdir()
    cfg=dict(base,argv=argv,environment=env,pins=pins,connected_shoulder_parent_launch=digest(job/'launch.json'),
             candidate_stage='native preparation; volume and earlier force owners retained as controls')
    if attachments:
        cfg['regional_tendon_manifest']=digest(tendon_path.with_suffix('.manifest.json'))
        cfg['pins'][str(tendon_path.with_suffix('.manifest.json'))]=cfg['regional_tendon_manifest']
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    inherited={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}
    result=execute_native(argv,base['cwd'],inherited|env,out,minimum_free_bytes=4*2**30,timeout_s=300)
    print(json.dumps(result));return result['exit']


if __name__=='__main__':sys.exit(run(*(Path(k).resolve() for k in sys.argv[1:])))
