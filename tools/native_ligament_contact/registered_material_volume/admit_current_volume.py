"""Use the production whole-volume constructor with the selected anatomical payloads."""
import hashlib,json,os,shutil,subprocess,sys,time,re
from pathlib import Path
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main(full,selected,binary,ram,output):
    a,b=[json.loads(p.read_text()) for p in (full,selected)]
    output.mkdir(exist_ok=True);src=full.parent
    if 'physical_volume_release_reference' in a:
        seed=a['physical_volume_release_reference'];source=Path(seed['launch_path'])
        if sha(source)!=seed['launch_sha256']:raise ValueError('Parent native source changed')
        src=source.parent
    def relocate(v):return v.replace(str(src),str(ram))
    argv=[relocate(v) for v in a['argv']];env={k:relocate(v) for k,v in a['environment'].items()}
    if not Path(argv[4]).is_relative_to(ram):raise ValueError('Output must be isolated to the trial volume')
    combined={}
    # The authored physical volume uses its recorded body mass, material and
    # bed ownership. The current anatomical input identity must pass its own gates.
    for flag in ['--skin-payload','--soft-tissue-payload','--torso-anatomy-payload','--resting-anatomy-receipt']:
        argv[argv.index(flag)+1]=b['argv'][b['argv'].index(flag)+1]
    argv[:4]=[str(binary),*b['argv'][1:4]]
    mode=os.environ.get('NUMI_SELECTED_TRIAL_MODE','source')
    if mode not in ('source','motion'):raise ValueError('Unknown selected-volume run mode')
    if mode=='source':
        argv[argv.index('--muscle-step-count')+1]='1'
    else:
        ticks=int(os.environ.get('NUMI_SELECTED_TRIAL_ROOTS','1'))
        if ticks<1:raise ValueError('Nonpositive physical horizon')
        argv[argv.index('--muscle-step-count')+1]=str(ticks)
        if '--mechanics-only' in argv:argv.remove('--mechanics-only')
    env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_SOURCE_ONLY']='1' if mode=='source' else '0'
    # Only the original whole-volume contact owner can contribute during admission.
    if env.get('NUMI_HUMAN_LOCAL_BED_LAYERS','0')!='0':raise ValueError('Duplicate condensed-volume authority')
    for k in ['NUMI_HUMAN_RESTING_SKIN_FAT_FEM','NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM','NUMI_HUMAN_RESTING_PASSIVE_FASCIA','NUMI_HUMAN_RESTING_AORTIC_WALL','NM_VASCULAR_STEP_EXTERNAL_PRESSURE']:
        if env.get(k)!='1':raise ValueError('Whole-owner baseline missing '+k)
    for path,digest in a['pins'].items():
        if sha(path)!=digest:raise ValueError('Full input changed '+path)
    for path in argv[:4]+[argv[argv.index(flag)+1] for flag in ['--skin-payload','--soft-tissue-payload','--torso-anatomy-payload','--resting-anatomy-receipt']]:
        pin=sha(path);expected=b['pins'].get(path)
        if expected is not None and expected!=pin:raise ValueError('Current input changed '+path)
        combined[path]=pin
    runtimeLib=env.get('NUMI_MATTER_METALLIB_OVERRIDE')
    if runtimeLib:combined[runtimeLib]=sha(runtimeLib)
    c=dict(argv=argv,environment=env,cwd=a['cwd'],pins={**a['pins'],**combined},selected_scene_receipt=sha(selected),full_scene_receipt=sha(full))
    (output/'launch.json').write_text(json.dumps(c,indent=2)+'\n')
    env={**os.environ,**env};started=time.monotonic();reason=None
    with (output/'stdout.log').open('w') as o,(output/'stderr.log').open('w') as e:
        proc=subprocess.Popen(argv,cwd=c['cwd'],env=env,stdout=o,stderr=e)
        while proc.poll() is None:
            if shutil.disk_usage(output).free<1024**3:reason='OS disk below 1 GiB'
            elif shutil.disk_usage(ram).free<100*1024**2:reason='temporary workspace below 100 MiB'
            elif time.monotonic()-started>900:reason='preparation exceeded 900 wall seconds'
            if reason:proc.terminate();proc.wait();break
            time.sleep(1)
    log=(output/'stdout.log').read_text();err=(output/'stderr.log').read_text()
    result=dict(exit_code=proc.returncode,wall_seconds=time.monotonic()-started,watchdog=reason,
        bindings=[x for x in log.splitlines() if any(k in x for k in ['registered_volume_native_preflight','registered_volume_typed_O_detail_bridge','whole_body_skin_fat_registered_volume_install','registered_volume_source_only','registered_volume_native_role','whole_body_skin_fat_direct','physical_skin_registered_volume_source_only'])],
        errors=[x for x in err.splitlines() if 'failed error=' in x],accepted_steps_reported=[int(n) for n in re.findall(r'(?<![a-z_])accepted_steps=(\d+)',log+'\n'+err)],source_only=(mode=='source'),requested_roots=int(argv[argv.index('--muscle-step-count')+1]))
    result['source_preparation_complete']='registered_volume_source_only=PASS' in log
    result['accepted_steps']=0 if mode=='source' else max(result['accepted_steps_reported'],default=None)
    native=Path(argv[4])
    if mode=='motion' and native.is_dir():
        for p in native.rglob('*'):
            if p.is_file() and p.suffix in ('.csv','.json','.log') and (p.stat().st_size<=300000 or p.name=='resting-ocular-sclera-window.json'):
                dest=output/'native'/p.relative_to(native);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        if proc.returncode==0 and os.environ.get('NUMI_SELECTED_RETAIN_TERMINAL_PACK')=='1':
            step=result['requested_roots']
            receipt=native/'accepted-geometry'/f'step-{step}.receipt.json'
            accepted=json.loads(receipt.read_text())
            pack=native/'accepted-geometry'/f'step-{step}.mrvpack'
            if accepted['accepted_step']!=step or Path(accepted['accepted_pack_path'])!=pack or sha(pack)!=accepted['pack_file_sha256']:
                raise ValueError('Terminal geometry does not match its native transaction')
            dst=output/'terminal.mrvpack.zst'
            subprocess.run(['/opt/homebrew/bin/zstd','-q','-1','-T1',str(pack),'-o',str(dst)],check=True)
            result['terminal_pack_sha256']=accepted['pack_file_sha256']
            result['terminal_archive_sha256']=sha(dst)
        result['native_artifacts']=[str(p.relative_to(native)) for p in native.rglob('*') if p.is_file()]
    (output/'admission.json').write_text(json.dumps(result,indent=2)+'\n')
    if proc.returncode:sys.exit(1)
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
