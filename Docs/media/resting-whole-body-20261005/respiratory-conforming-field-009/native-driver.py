from pathlib import Path
import csv,datetime,hashlib,json,os,subprocess,sys,time
root=Path('/Users/n/numi-human-resting-evidence-20261005')
source=Path('/Users/n/numi-human-resting-conforming-source-009')
build=Path('/Users/n/numi-human-resting-conforming-build-009')
name=sys.argv[1];count=int(sys.argv[2]);inputs=Path(sys.argv[3]);captures=sys.argv[4]
out=root/name
out.mkdir(exist_ok=False)
old=json.loads((root/'cardiac-wall-native-cycle-001/invocation.json').read_text())
argv=old['argv'].copy();argv[0]=str(build/'bin/numi-human-native');argv[4]=str(out)
argv[argv.index('--muscle-step-count')+1]=str(count)
argv[argv.index('--resting-scene')+2]=str(root/'thorax-conforming-field-009/resting-reference-respiration.json')
argv[argv.index('--torso-anatomy-payload')+1]=str(inputs/'resting-thorax.nhanatomy')
argv[argv.index('--resting-anatomy-receipt')+1]=str(inputs/'resting-anatomy-receipt.json')
argv[argv.index('--resting-movie')+1]=str(out/'native-viewer.mov')
env=old['environment'].copy();env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']=captures
env["NUMI_HUMAN_GPU_TIMING"]="1";env["NUMI_HUMAN_GPU_TIMING_STAGE"]="cycle"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=[Path(x) for x in argv if Path(x).is_file()]
paths += [build/x for x in ['lib/libmetalrobo.dylib','shaders/MetalRobo.metallib','shaders/MetalRoboHyperPolicy.metallib','shaders/NumiNeuron.metallib','matter/shaders/HumanRespiration.metallib','matter/shaders/NumiMatter.metallib','matter/shaders/NumiMatterPhysicalStateDigest.metallib','CMakeCache.txt']]
assets={str(p):sha(p) for p in paths}
files=['apps/NumiHumanRestingAnatomy.hpp','apps/NumiHumanRestingVisual.hpp','apps/NumiHumanRestingWindow.hpp','apps/numilab_human_myosim_visual_probe.mm','include/metalrobo/numi_human_resting_visual_gpu.h','matter/src/human_respiration.metal','matter/tools/cardiac_geometry_binding.py','matter/tools/resting_intervention_study.py','matter/tools/test_resting_intervention_study.py']
source_sha={x:sha(source/x) for x in files}
(out/'source.diff').write_bytes(subprocess.check_output(['git','-C',str(source),'diff','--binary']))
invocation={'argv':argv,'environment':env,'asset_sha256':assets,'source_revision':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),'source_file_sha256':source_sha,'source_diff_sha256':sha(out/'source.diff'),'driver_sha256':sha(Path(__file__)),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'qualification':'Engineering test of bounded short-edge respiratory preparation plus complete integrated GPU stage profiling. Uses the repaired source-bound cardiac wall; abdominal registration and full anatomical acceptance remain pending.'}
(out/'invocation.json').write_text(json.dumps(invocation,indent=2)+'\n')
t=time.monotonic()
with (out/'native.log').open('wb') as log:
 result=subprocess.run(argv,env={**os.environ,**env},stdout=log,stderr=subprocess.STDOUT)
changed=[str(p) for p in paths if sha(p)!=assets[str(p)]]+[str(source/p) for p in files if sha(source/p)!=source_sha[p]]
record={'returncode':result.returncode,'wrapper_wall_seconds':time.monotonic()-t,'changed_sources':changed,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
if (out/'resting-surface-audit.csv').is_file():
 rows=list(csv.DictReader((out/'resting-surface-audit.csv').open()))
 if rows:record['last_surface_frame']=rows[-1]
(out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
