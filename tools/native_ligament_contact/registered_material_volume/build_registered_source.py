"""Compile the native support seed evaluator with its source-matched physical parent."""
from pathlib import Path
import json,sys,subprocess,shutil
from emit_skin_source_registration import sha
source,reg,out,app=map(Path,sys.argv[1:]);out.mkdir(exist_ok=True)
report=json.loads((reg/'physical-parent-identity.json').read_text())
h=reg/'NumiHumanSelectedSkinReference.hpp'
if sha(h)!=report['header_sha256']:raise ValueError('Unpinned source registration')
for p in source.iterdir():
    if p.is_file() and p.suffix in ('.hpp','.metalinc','.metal') and not (out/p.name).exists():(out/p.name).hardlink_to(p)
(out/'metalrobo').mkdir(exist_ok=True)
for f in (source/'metalrobo').glob('*.metallib'):
    if not (out/'metalrobo'/f.name).exists():(out/'metalrobo'/f.name).hardlink_to(f)
shutil.copyfile(h,out/h.name)
name='numilab_human_myosim_visual_probe.mm';s=(source/name).read_text()
a=s.index('bool evaluateRegisteredWholeBodySupportOuter(');b=s.index('std::unique_ptr<PassiveFasciaRuntimeOwner>',a);block=s[a:b]
start='    if(!makeRegisteredSkinMechanicalRegion('
if block.count(start)!=1:raise ValueError('Unrecognized native support evaluator')
block=block.replace(start,'''    const bool rebound=payloadSHA256==selected_skin_reference::payload;
    if(rebound) {
        const char* full=std::getenv("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_SOURCE_ONLY");
        if(!full||std::strcmp(full,"1")!=0){error="selected physical skin requires its full source volume";return false;}
    }
'''+start)
for old,replacement in [('kWholeBodySkinCandidateSourcePayloadSHA256','(rebound?selected_skin_reference::payload:kWholeBodySkinCandidateSourcePayloadSHA256)'),('kWholeBodySkinSourceEmbeddingResidualWorldM','(rebound?selected_skin_reference::residual:kWholeBodySkinSourceEmbeddingResidualWorldM)'),('kWholeBodySkinDonorFrames','(rebound?selected_skin_reference::frames:kWholeBodySkinDonorFrames)'),('kWholeBodySkinCandidateReferenceWorldPoints','(rebound?selected_skin_reference::points:kWholeBodySkinCandidateReferenceWorldPoints)')]:
    if block.count(old)!=1:raise ValueError(old)
    block=block.replace(old,replacement)
(out/name).write_text(s[:a]+'#include "NumiHumanSelectedSkinReference.hpp"\n'+block+s[b:])
rows=json.loads((source/'build-commands.json').read_text());host=[[arg.replace(str(source)+'/',str(out)+'/') for arg in row] for row in rows if row[0].endswith('clang++')]
if len(host)!=2:raise ValueError('Expected separate native compile and link commands')
(out/'build-commands.json').write_text(json.dumps(host,indent=2)+'\n')
for cmd in host:subprocess.run(cmd,cwd=app/'build-native',check=True)
(out/'identity.json').write_text(json.dumps({str(f):sha(f) for f in [source/name,out/name,h,out/'numi-human-native',*(out/'metalrobo').glob('*.metallib')]},indent=2)+'\n')
