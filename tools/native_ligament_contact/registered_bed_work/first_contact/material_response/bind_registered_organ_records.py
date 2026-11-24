"""Bind intact source records across the shoulder-only anatomy registration change."""
from pathlib import Path
import sys,struct,json,hashlib,re,os,subprocess
import numpy as np
from compile_connected_joint import digest
ORIGINAL='3b0acb6146358f3302f2b85e2214bafa9f2fae7b0e51af70fd0838d6609da4a4'
def inspect(path):
    raw=path.read_bytes();_,_,n,v,i,_,_=struct.unpack_from('<8s5I32s',raw);vbase=60+32*n;ibase=vbase+24*v;assert ibase+4*i==len(raw)
    rec=np.frombuffer(raw,'<u4',8*n,60).reshape(-1,8);r={}
    for stable in [4,5,13,322,463]:
        hit=rec[rec[:,5]==stable];assert len(hit)==1;b,fv,n,fi,ic,s,l,res=map(int,hit[0]);vs=raw[vbase+fv*24:vbase+(fv+n)*24];idx=np.frombuffer(raw,'<u4',ic,ibase+fi*4)
        assert ic%3==0 and idx.min()>=fv and idx.max()<fv+n and b==(23 if s==322 else 20)
        local=(idx-fv).astype('<u4').tobytes();r[stable]=(struct.pack('<5I',s,b,l,n,ic)+vs+local,hashlib.sha256(vs+idx.tobytes()).hexdigest(),(b,l,n,ic//3,fv,fi))
    return r
def run(base,scene,source,out):
    if digest(source)!=ORIGINAL:raise ValueError('Reference payload is not the originally admitted one')
    cfg=json.loads((scene/'launch.json').read_bytes());nha=Path(cfg['argv'][cfg['argv'].index('--torso-anatomy-payload')+1]);pin=digest(nha)
    if cfg['pins'][str(nha)]!=pin:raise ValueError('Anatomy differs from scene pin')
    a=inspect(source);b=inspect(nha);report={}
    for stable in a:
        if a[stable][0]!=b[stable][0]:raise ValueError(f'Material source record changed at ID {stable}')
        report[stable]=dict(before=a[stable][2],current=b[stable][2],typed_geometry_sha256=hashlib.sha256(a[stable][0]).hexdigest(),old_buffer_sha256=a[stable][1],new_buffer_sha256=b[stable][1])
    out.mkdir();app=out/'app';app.mkdir();src=base/'app';unit='numilab_human_myosim_visual_probe.mm'
    for f in src.iterdir():
        if f.name!=unit and f.suffix in ('.mm','.hpp','.inc','.h'):os.link(f.resolve(),app/f.name)
    s=(src/unit).read_text()
    def replace(old,new):
        nonlocal s
        if s.count(old)!=1:raise ValueError('Ambiguous native scene binding: '+old[:100])
        s=s.replace(old,new)
    replace('payloadDigest==prostatePairedBone49PayloadSHA256;',f'payloadDigest==prostatePairedBone49PayloadSHA256||payloadDigest=="{pin}";')
    # Keep the compiled original FEM asset identities. The active NHA may have
    # different firstVertex and firstIndex offsets; its actual buffer digest is
    # compared independently after the body/layer/record eligibility checks.
    replace('require(payloadDigest==expectedPayload,',f'const bool registeredSourceRecords=(payloadDigest=="{pin}");\n        require(payloadDigest==expectedPayload||registeredSourceRecords,')
    replace('require(asset&&asset->payloadSHA256==payloadDigest,','require(asset&&asset->payloadSHA256==expectedPayload,')
    newids='\n'.join(f'                case {i}u: require(asset->meshSHA256=="{a[i][1]}","source organ asset differs"); expectedMesh="{b[i][1]}"; break;' for i in (4,5,13))
    old='require(meshDigest==asset->meshSHA256,'
    replace(old,'std::string_view expectedMesh=asset->meshSHA256;\n            if(registeredSourceRecords){switch(asset->stableID){\n'+newids+'\n                default: throw std::runtime_error("unregistered physical organ");\n            }}\n            require(meshDigest==expectedMesh,')
    for i in (4,5,13):
        header=(src/'NumiHumanRegisteredOrganBatchAssets.hpp').read_text()
        if a[i][1] not in header:raise ValueError('Paired asset uses a different record hash')
    replace('require(payloadSHA==tetgen_assets::kActiveNHAPayloadSHA256,',f'require(payloadSHA==tetgen_assets::kActiveNHAPayloadSHA256||payloadSHA=="{pin}",')
    (app/unit).write_text(s);(out/'records.json').write_text(json.dumps(dict(source=ORIGINAL,current=pin,source_path=str(source),current_path=str(nha),records=report),indent=2)+'\n')
    cmds=[[v.replace(str(src)+'/',str(app)+'/') for v in c] for c in json.loads((base/'build.json').read_bytes())['commands']]
    with (out/'build.log').open('w') as log:
        for c in cmds:subprocess.run(c,check=True,stdout=log,stderr=subprocess.STDOUT)
    (out/'build.json').write_text(json.dumps(dict(commands=cmds,source=digest(src/unit),result=digest(app/unit),binary=digest(app/'numi-human-native'),verified_records=digest(out/'records.json')),indent=2)+'\n')
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
