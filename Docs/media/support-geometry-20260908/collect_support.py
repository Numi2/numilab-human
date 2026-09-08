from pathlib import Path
import datetime,hashlib,json,subprocess
out=Path('/Users/n/numi-human-balance-solver-20260908')
repo=Path('/Users/n/MetalRobo-human-completion-20260907');build=Path('/Users/n/MetalRobo-human-completion-build-20260907')
brain=Path('/Users/n/numi-brain-human-completion-20260907')
def sh(args,cwd=None):return subprocess.check_output(args,cwd=cwd,text=True).strip()
def item(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
source=['CMakeLists.txt','include/metalrobo/NumiHumanMuscleEquilibrium.hpp','src/core/NumiHumanMuscleEquilibrium.cpp','src/metal/MetalNumanXHumanIO.mm','apps/numilab_human_myosim_visual_probe.mm','tests/numi_human_static_support_test.cpp']
bins=['lib/libmetalrobo.dylib','bin/metalrobo_numilab_human_myosim_visual_probe','bin/metalrobo_numilab_human_myosim_reference_probe','bin/metalrobo_numi_human_static_support_test','bin/metalrobo_numanx_human_io_probe','shaders/MetalRobo.metallib','matter/shaders/NumiMatter.metallib']
inputs=set()
for row in json.loads((out/'qualified.json').read_text()):
 inputs.update(arg for arg in row['command'][1:] if Path(arg).is_file())
for val in json.loads((out/'costal-regression-environment.json').read_text()).values():
 if Path(val).is_file():inputs.add(val)
meta={'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'native_revision':sh(['git','rev-parse','HEAD'],repo),'native_status':sh(['git','status','--porcelain'],repo),'brain_revision':sh(['git','rev-parse','HEAD'],brain),'brain_status':sh(['git','status','--porcelain'],brain),'toolchain':{'macos':sh(['sw_vers']),'xcode':sh(['xcodebuild','-version']),'swift':sh(['swift','--version']),'hardware':sh(['sysctl','-n','machdep.cpu.brand_string'])},'native_sources':{p:item(repo/p) for p in source},'binaries':{p:item(build/p) for p in bins},'inputs':{str(p):item(Path(p)) for p in sorted(inputs)},'brain_test':item(brain/'.build/arm64-apple-macosx/release/NumiBrainPackageTests.xctest/Contents/MacOS/NumiBrainPackageTests')}
require_clean = meta['native_status']=='' and meta['brain_status']==''
assert require_clean,meta
(out/'final-stack.json').write_text(json.dumps(meta,indent=2)+'\n')
