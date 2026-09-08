from pathlib import Path
import hashlib,json,subprocess,platform
out=Path('/Users/n/numi-human-active-locomotion-20260908')
roots={'numi-lab':Path('/Users/n/MetalRobo-human-completion-20260907'),'numi-brain':Path('/Users/n/numi-brain-human-completion-20260907')}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(p,*args):return subprocess.check_output(['git','-C',str(p),*args],text=True).strip()
build=Path('/Users/n/MetalRobo-human-completion-build-20260907');brain=roots['numi-brain']
files={'native_library':build/'lib/libmetalrobo.dylib','native_metallib':build/'shaders/MetalRobo.metallib','matter_metallib':build/'matter/shaders/NumiMatter.metallib','brain_cli':brain/'.build/arm64-apple-macosx/release/numi-brain-gate-c','brain_test':brain/'.build/arm64-apple-macosx/release/NumiBrainPackageTests.xctest/Contents/MacOS/NumiBrainPackageTests','mlx_metallib':brain/'.build/arm64-apple-macosx/release/mlx.metallib'}
receipt={'format':'numi-active-muscle-evidence-v1','promotable':False,'host':platform.node(),'repositories':{name:{'commit':git(p,'rev-parse','HEAD'),'status':git(p,'status','--porcelain')} for name,p in roots.items()},'toolchain':{cmd:subprocess.check_output(cmd.split(),text=True).strip() for cmd in ['sw_vers','xcodebuild -version','swift --version']},'binaries':{key:{'path':str(p),'sha256':sha(p)} for key,p in files.items()},'files':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.suffix in ['.log','.json','.py'] and p.name!='receipt.json'}}
for mode in ['active','replay','unavailable','emergency']:
 p=out/('costal-'+mode)/'muscle-locomotor-research.json'
 if p.exists():receipt.setdefault('costal_captures',{})[mode]=json.loads(p.read_text())
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt['repositories'],indent=2))
