"""Extend the current owner's terminal observer, using its already accepted GPU snapshot."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
prior,folder,app=map(Path,sys.argv[1:])
b=folder/'build';b.mkdir(exist_ok=True,parents=True)
for dst,src in [('libmetalrobo.dylib',prior/'build/libmetalrobo.dylib'),('NumiMatter.metallib',prior/'build/NumiMatter.metallib'),('metalrobo/HumanRestingSkin.metallib',prior/'build/metalrobo/HumanRestingSkin.metallib'),('metalrobo/MetalRobo.metallib',prior/'build/metalrobo/MetalRobo.metallib')]:
    path=b/dst;path.parent.mkdir(exist_ok=True)
    if not path.exists():path.hardlink_to(src)
for name in ('run_scene.py','strip.json'):shutil.copyfile(prior/name,folder/name)
cmd=json.loads((prior/'build-commands.json').read_text());source=Path(cmd[0][-1]);text=source.read_text()
def edit(old,new):
    global text
    if text.count(old)!=1:raise ValueError(old)
    text=text.replace(old,new)
edit('#include "NumiHumanRegisteredLigaments.hpp"','#include "NumiHumanRegisteredLigaments.hpp"\n#include "NumiHumanTissueMassMoment.hpp"')
edit('        for(unsigned i=0;i<a.femNodes.size();++i) {','''        NumiHumanTissueMassMoment initialMassMoment,acceptedMassMoment;
        for(unsigned i=0;i<a.femNodes.size();++i) {
            initialMassMoment.add(z.femNodes[i]);
            acceptedMassMoment.add(a.femNodes[i]);''')
edit('        require(clockMatches && passesMatch,','''        require(initialMassMoment.massKg>0.0 && initialMassMoment.massKg==acceptedMassMoment.massKg,
                "native tissue mass changed between initial and accepted states");
        std::cout << std::setprecision(17) << "resting_full_fem_mass_moment nodes=" << a.femNodes.size()
                  << " mass_kg=" << acceptedMassMoment.massKg
                  << " initial_kg_m=" << initialMassMoment.firstKgM[0] << "," << initialMassMoment.firstKgM[1] << "," << initialMassMoment.firstKgM[2]
                  << " accepted_kg_m=" << acceptedMassMoment.firstKgM[0] << "," << acceptedMassMoment.firstKgM[1] << "," << acceptedMassMoment.firstKgM[2]
                  << " accepted_step=" << stepCount << std::endl;
        require(clockMatches && passesMatch,''')
(b/source.name).write_text(text);shutil.copyfile(Path(__file__).with_name('NumiHumanTissueMassMoment.hpp'),b/'NumiHumanTissueMassMoment.hpp')
def update(s):return str(b/source.name) if s==str(source) else s.replace(str(prior/'build/'),str(b)+'/')
cmd=[[update(k) for k in v] for v in cmd]
cmd[0][1:1]=['-iquote',str(source.parent)]
(folder/'build-commands.json').write_text(json.dumps(cmd,indent=2)+'\n')
for args in cmd:subprocess.run(args,cwd=app/'build-native',check=True)
(folder/'identities.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,b/source.name,b/'NumiHumanTissueMassMoment.hpp',b/'numi-human-native',b/'NumiMatter.metallib',b/'metalrobo/HumanRestingSkin.metallib',b/'libmetalrobo.dylib')},indent=2)+'\n')
