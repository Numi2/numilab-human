"""Compile the existing native catalog against the independently checked candidate material identity."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(d,project,source,bone,launch,sites,changes,output):
    base=d/'candidate'; header=(base/'NumiHumanRegisteredLigaments.hpp').read_text()
    doc=json.loads(bone.with_suffix('.manifest.json').read_text())
    pin=digest(bone);assert doc['payload']['sha256']==pin
    prior=json.loads(sites.read_text());basepin=doc['source']['connected_girdle_parent']['bone_sha256']
    assert basepin==prior['bone_payload_sha256']
    oldparts=re.search(r'(kReferenceBonesPayloadSha256\{\{)(.*?)(\}\};)',header,re.S)
    assert bytes(int(v,16) for v in re.findall(r'0x([a-f0-9]{2})',oldparts[2])).hex()==basepin
    newparts=','.join(f'0x{x:02x}' for x in bytes.fromhex(pin))
    header=header[:oldparts.start(2)]+newparts+header[oldparts.end(2):]
    header=header.replace('v6-lumbar-interspinous-bones-'+basepin[:12],'v7-shoulder-candidate-bones-'+pin[:12])
    if changes is not None:
        data=json.loads(changes.read_text())
        assert data['candidate_sha256']==pin
        mark='  v.push_back(spec);'; assert header.count(mark)==1
        extra=[]
        for row in data['rows']:
            details=[]
            for pt in row['attachments']:
                weights=pt['barycentric']; assert len(weights)==3 and all(0<w<1 for w in weights) and abs(sum(weights)-1)<1e-13
                details.append('MaterialAttachment{'+str(pt['triangle'])+', {'+','.join(str(x) for x in weights)+'}}')
            extra.append('  if (n=="'+row['name']+'" && side=='+str(row['side'])+') {spec.materialAttachments={'+','.join(details)+'}; spec.seedLateralityMirrored=false;spec.geometrySource="'+row['source']+'";}')
        header=header.replace(mark,'\n'.join(extra)+'\n'+mark)
    output.mkdir(exist_ok=True)
    (output/'NumiHumanRegisteredLigaments.hpp').write_text(header)
    code=(base/'cook_checked.cpp').read_text()
    # Keep production hash handling. The checker derives the SHA directly from
    # the file it baked, then checks each material source against the catalog.
    code=code.replace('#include <fstream>','#include <fstream>\n#include <CommonCrypto/CommonDigest.h>')
    mark='    // Input SHA-256 is independently verified by the invoking driver.'
    assert mark in code
    code=code.replace(mark,'''    std::ifstream bytes(argv[1], std::ios::binary);
    std::vector<unsigned char> raw((std::istreambuf_iterator<char>(bytes)), {});
    std::array<unsigned char, 32> sourceHash{};
    if (raw.empty() || raw.size() > UINT32_MAX) return 7;
    CC_SHA256(raw.data(), static_cast<CC_LONG>(raw.size()), sourceHash.data());
    if (sourceHash != lig::kReferenceBonesPayloadSha256) return 8;''')
    code=code.replace('poses.at(bank.bodyIndex), lig::kReferenceBonesPayloadSha256)', 'poses.at(bank.bodyIndex), sourceHash)')
    (output/'cook_checked.cpp').write_text(code)
    host=Path(source)
    cmd=['/usr/bin/clang++','-std=c++23','-O2','-I'+str(project/'floor-scene-build'),'-I'+str(host/'include'),'-I'+str(host/'matter/include'),'-I'+str(host/'matter/tools'),str(output/'cook_checked.cpp'),'-o',str(output/'cook_catalog')]
    subprocess.run(cmd,check=True);cfg=json.loads(launch.read_text())
    rigid=Path(cfg['argv'][1]); assert cfg['pins'][str(rigid)]==digest(rigid)
    reports=[]
    for i,p in enumerate(prior['native_pose_inputs']):
        f=Path(p['path']); assert digest(f)==p['sha256']
        pose=next(json.loads(s.partition('=')[2]) for s in f.open() if s.startswith('registered_ligament_actual_pose_json='))
        assert pose['bonePayloadSha256']==basepin
        text='\n'.join(' '.join(map(str,v)) for v in pose['bodyPoses'])
        sub=subprocess.run([output/'cook_catalog',bone,rigid],capture_output=True,text=True,input=text)
        (output/f'catalog-{i}.csv').write_text(sub.stdout)
        reports.append(dict(entrance=i,passed=len(sub.stdout.splitlines())-1,returncode=sub.returncode,errors=sub.stderr.splitlines()))
    (output/'result.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(reports)
if __name__=='__main__':
    import sys
    d,p,s,b,l,a,c,o=sys.argv[1:]
    run(*map(Path,(d,p,s,b,l,a)),None if c=='-' else Path(c),Path(o))
