"""Build an isolated diagnostic twin from the actual launch's native runtime and application."""
from pathlib import Path
import argparse,json,subprocess,concurrent.futures
from compile_connected_joint import digest
def run(base,prod,out):
    out.mkdir(exist_ok=True)
    read=lambda f:json.loads(f.read_bytes());parent=read(base/'build.json')
    def call(cmd,path):
        with path.open('w')as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    here=Path(__file__).resolve().parent
    for cmd in [['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-c',str(here/'certificate_rows.metal'),'-o',str(out/'rows.air')],
                ['xcrun','-sdk','macosx','metallib',str(out/'rows.air'),'-o',str(out/'rows.metallib')]]:
        call(cmd,out/'shader.log')
    commands=parent['commands']
    previous=read(prod/'build.json')
    substitutions={}
    for folder,file,idx in [('runtime','runtime.mm',0),('app','app-runtime.mm',2)]:
        root=out/folder;root.mkdir(exist_ok=True);src=base/folder/file
        program=src.read_text(); mark='                dispatchThreads("nm_mixed_capture_contact_certificate_witness",environments,[&] {'
        if program.count(mark)!=1:raise ValueError('Unmatched certificate owner')
        (root/file).write_text(program.replace(mark,(here/'observe.inc').read_text()+mark))
        for p in src.parent.glob('*.inc'):
            dst=root/p.name
            if not dst.exists():dst.symlink_to(p)
        for name in (file,file+'.o'):
            substitutions[str(base/folder/name)]=str(root/name)
    substitutions[str(base/'runtime/libmetalrobo.dylib')]=str(out/'runtime/libmetalrobo.dylib')
    substitutions[str(prod/'app/numi-human-native')]=str(out/'app/numi-human-native')
    cmds={i:[substitutions.get(s,s) for s in (previous['commands'][-1] if i==4 else commands[i])] for i in (0,1,2,4)}
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        jobs=[ex.submit(call,cmds[i],out/f'compile-{i}.log')for i in (0,2)]
        for job in jobs:job.result()
    for i in (1,4):call(cmds[i],out/f'compile-{i}.log')
    (out/'pins.json').write_text(json.dumps({str(p):digest(p) for p in [out/'runtime/libmetalrobo.dylib',out/'app/numi-human-native',out/'rows.metallib']},indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('base','prod','out'):p.add_argument(n,type=lambda v:Path(v).resolve())
    run(**vars(p.parse_args()))
