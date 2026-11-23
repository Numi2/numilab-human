"""Run the connected Human with prescribed tissue and skeleton starting at the same GPU pose."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

# The Human repository owns the bounded native runner in the digit-material tools.
runtime_tools=Path(__file__).resolve().parents[3]/'registered_digit_material'
if runtime_tools.is_dir():
    sys.path.insert(0,str(runtime_tools))
from run_cavity_motion import digest as sha, execute_native


def replace_unique(text, old, new):
    if text.count(old)!=1:
        raise ValueError('Native source boundary is ambiguous: '+old[:80])
    return text.replace(old,new)


def build_native(base, out):
    receipt=json.loads((base/'build.json').read_bytes())
    if sha(base/'numi-human-native')!=receipt['executable']['sha256']:
        raise ValueError('Control executable changed')
    out.mkdir()
    main='numilab_human_myosim_visual_probe.mm'
    for source in base.iterdir():
        if source.name!=main and source.suffix in {'.mm','.metal','.metalinc','.inc','.hpp','.h'}:
            os.link(source,out/source.name)
    header=Path(__file__).with_name('NumiHumanRegisteredAttachmentRelease.hpp')
    (out/header.name).write_bytes(header.read_bytes())
    code=replace_unique((base/main).read_text(),'int main(int argc, char** argv) {',
        '#include "NumiHumanRegisteredAttachmentRelease.hpp"\n\nint main(int argc, char** argv) {')
    path_decl='            std::filesystem::path selectedMatterMetallib ='
    code=replace_unique(code,path_decl,'            RegisteredAttachmentRelease attachmentRelease;\n'+path_decl)
    snap='                    transaction.initial=coupled.physiology.runtime.snapshot();'
    code=replace_unique(code,snap,snap+'''
                    if(!attachmentRelease.ids.empty())
                        installRegisteredAttachmentRelease(attachmentRelease,transaction.initial,coupled.physiology.runtime);''')
    marker='                    overriddenPoseQ=initialTissueQ;poseQ=overriddenPoseQ;'
    code=replace_unique(code,marker,marker+'''
                    if (const char* align=std::getenv("NUMI_HUMAN_REGISTERED_ATTACHMENT_RELEASE")) {
                        require(std::string_view(align)=="1","Expected attachment release mode 1");
                        attachmentRelease=prepareRegisteredAttachmentRelease(rigid.model,poseQ,*restingPassiveFasciaOwner,
                                                               selectedMatterMetallib,appendSource);
                    }''')
    (out/main).write_text(code)
    commands=[]
    for original in receipt['commands']:
        cmd=[v.replace(str(base)+'/',str(out)+'/') if v.startswith(str(base)+'/') else v for v in original]
        subprocess.run(cmd,check=True)
        commands.append(cmd)
    result=dict(parent_executable=receipt['executable'],parent_build_sha256=sha(base/'build.json'),
                library=receipt['library'],commands=commands,header_sha256=sha(header),
                application_sha256=sha(out/main),executable_sha256=sha(out/'numi-human-native'))
    (out/'build.json').write_text(json.dumps(result,indent=2)+'\n')


def run(build, control, dest, split, apply=True, capture_terminal=False):
    j=json.loads((control/'launch.json').read_bytes())
    receipt=json.loads((build/'build.json').read_bytes())
    if sha(Path(j['argv'][0]))!=receipt['parent_executable']['sha256'] or sha(build/'numi-human-native')!=receipt['executable_sha256']:
        raise ValueError('Mismatched control or candidate identity')
    argv=[s.replace(str(control)+'/',str(dest)+'/') for s in j['argv']]
    env={k:v.replace(str(control)+'/',str(dest)+'/') for k,v in j['environment'].items()}
    argv[0]=str(build/'numi-human-native')
    i=argv.index('--muscle-step-seconds')+1
    n=argv.index('--muscle-step-count')+1
    if argv[n]!='1' or split<1:
        raise ValueError('Control must be a single physical root')
    argv[i]=str(float(argv[i])/split)
    argv[n]=str(split)
    if capture_terminal:
        env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']=str(split)
    else:
        env.pop('NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS',None)
    env['NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP']=str(split)
    if apply:env['NUMI_HUMAN_REGISTERED_ATTACHMENT_RELEASE']='1'
    dest.mkdir()
    (dest/'native-traces').mkdir()
    (dest/'launch.json').write_text(json.dumps(dict(j,argv=argv,environment=env,
        original_launch_sha256=sha(control/'launch.json'),registered_release_build_sha256=sha(build/'build.json'),
        registered_release_enabled=apply,subdivision=split),indent=2)+'\n')
    inherited={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}
    result=execute_native(argv,j['cwd'],inherited|env,dest,minimum_free_bytes=4*2**30,timeout_s=600)
    print(json.dumps(result),flush=True)
    return result['exit']


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('base',type=Path)
    cli.add_argument('control',type=Path)
    cli.add_argument('build',type=Path)
    cli.add_argument('output',type=Path)
    cli.add_argument('--compile',action='store_true')
    cli.add_argument('--capture-terminal',action='store_true')
    cli.add_argument('--control',action='store_true',dest='disable')
    cli.add_argument('--subdivide',type=int,default=1)
    a=cli.parse_args()
    if a.compile:build_native(a.base.resolve(),a.build.resolve())
    raise SystemExit(run(a.build.resolve(),a.control.resolve(),a.output.resolve(),a.subdivide,not a.disable,a.capture_terminal))
