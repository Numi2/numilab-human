"""Register the existing seven-coordinate field after proving every referenced surface is unchanged."""
from pathlib import Path
import json,sys,os,subprocess
from compile_connected_joint import digest
from run_cavity_motion import execute_native

def run(job,base,proof,out):
    rd=lambda p:json.loads(p.read_bytes());cfg=rd(base/'launch.json');a=cfg['argv'];receipt=Path(a[a.index('--resting-anatomy-receipt')+1]);nha=Path(a[a.index('--torso-anatomy-payload')+1]);d=rd(proof);surf=d['ranges']
    if digest(receipt)!=d['receipt'] or cfg['pins'][str(receipt)]!=d['receipt'] or digest(nha)!=d['current'] or len(surf)!=152 or sum(v['physical'] for v in surf)!=7 or any(v['changed_vertices'] or not v['triangle_parity'] for v in surf):raise ValueError('Cardiac source domain differs')
    r=rd(receipt);field=r['provenance']['cardiac_geometry_binding']['common_field']
    if field['anatomy_payload_sha256']!=d['original']:raise ValueError('Incorrect original field')
    field['anatomy_payload_sha256']=d['current'];out.mkdir();p=out/'resting-anatomy-receipt.json'
    # The released receipt already uses the current anatomy for the rib labels.
    # Edit the single remaining historical field link in a copy-on-write clone.
    template=Path(rd(job/'current-owner-scene/launch.json')['argv'][a.index('--resting-anatomy-receipt')+1])
    import re
    raw=template.read_bytes();expected=rd(template);ribProof=rd(base/'intercostal-registration.json')
    edits=[]
    for row in ribProof['bindings']:
        name=row['binding'];pos=raw.index(('"'+name+'"').encode());n=expected['functional_bindings'][name]
        for key,old,new in [('anatomy_payload_sha256',d['original'],d['current']),('bones_payload_sha256',ribProof['original_bones'],ribProof['current_bones'])]:
            if n[key]!=old:raise ValueError('Unexpected rib parent')
            match=re.search(b'"'+key.encode()+b'"\\s*:\\s*"('+old.encode()+b')"',raw[pos:])
            if not match:raise ValueError('Rib field pin absent')
            edits.append((pos+match.start(1),old,new));n[key]=new
    inCommon=raw.index(b'"common_field"',raw.index(b'"cardiac_geometry_binding"'))
    match=re.search(b'"anatomy_payload_sha256"\\s*:\\s*"('+d['original'].encode()+b')"',raw[inCommon:])
    if not match:raise ValueError('Common field parent pin absent')
    edits.append((inCommon+match.start(1),d['original'],d['current']))
    expected['provenance']['cardiac_geometry_binding']['common_field']['anatomy_payload_sha256']=d['current']
    if expected!=r or len(set(v[0] for v in edits))!=7:raise ValueError('Unexpected unrelated receipt changes')
    subprocess.run(['/bin/cp','-c',str(template),str(p)],check=True)
    with p.open('r+b') as file:
        for offset,old,new in edits:
            file.seek(offset)
            if file.read(64)!=old.encode():raise ValueError('Receipt byte span differs')
            file.seek(offset);file.write(new.encode())
    if rd(p)!=r:raise ValueError('Cloned receipt has unexpected edits')
    for k,v in d['sidecars'].items():
        f=Path(v['path']);rel=Path(v['name'])
        if rel.is_absolute() or '..' in rel.parts or digest(f)!=field[k]['sha256'] or digest(f)!=v['sha256']:raise ValueError('Incorrect source map')
        target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);os.link(f,target);cfg['pins'][str(target)]=v['sha256']
    original=base/'launch.json';a[a.index('--resting-anatomy-receipt')+1]=str(p);a[4]=str(out/'native-traces')
    cfg['parent_launch']=digest(original);cfg['pins'][str(p)]=digest(p);env={k:v.replace(str(base)+'/',str(out)+'/') for k,v in cfg['environment'].items()};cfg['environment']=env
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(execute_native(a,cfg['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|env,out,minimum_free_bytes=4*2**30,timeout_s=400),flush=True)
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
