"""Carry byte-identical tendon laws on unchanged bones while recooking the four registered members."""
import hashlib,json,struct,sys
from pathlib import Path
from compare_tendons import compare
BNAME='bodyparts3d-myosim-major-bones.nhbones'
TNAME='numi-human-tendon-attachments.nhtendon'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def unpack(p):
    d=json.loads(p.with_suffix('.manifest.json').read_text());assert d['payload']['sha256']==sha(p)
    buf=p.read_bytes(); h=struct.unpack_from('<8s10I32s32s32s',buf)
    assert h[0]==b'NHTEND3\0' and h[1]==3 and h[5]==len(d['endpoints']) and len(buf)==144+64*h[5]+288*h[6]
    raw=[];seen=set()
    for i,record in enumerate(d['endpoints']):
        start=144+i*64;r=bytearray(buf[start:start+64]);index=struct.unpack_from('<I',r,24)[0]
        assert struct.unpack_from('<I',r,16)[0]==record['body_index']
        if index==0xffffffff: env=b'';assert record['surface'] is None
        else:
            offset=144+h[5]*64+288*index
            assert index<h[6] and index not in seen and record['surface'] is not None
            seen.add(index);env=buf[offset:offset+288]
            struct.pack_into('<I',r,24,0)
        raw.append((bytes(r),env))
    assert len(seen)==h[6]
    return d,h,raw
def run(launch,bones,regional0,regional1,out):
    cfg=json.loads(launch.read_text());old=Path(cfg['argv'][3]);new=bones/BNAME
    td=Path(cfg['argv'][cfg['argv'].index('--tendon-payload')+1]);assert cfg['pins'][str(td)]==sha(td)
    p,q,r=[unpack(v) for v in (td,regional0/TNAME,regional1/TNAME)]
    # Verify the entire rest of the native model byte-for-byte. The header
    # discriminator and exactly four seven-component rigid poses may differ.
    a,b=old.read_bytes(),new.read_bytes()
    assert len(a)==len(b) and a[:24]==b[:24] and a[28:60]==b[28:60]
    change={8,9,10,11}; owners={};controls=[]
    for i in range(p[1][7]):
        start=60+60*i; aa,bb=a[start:start+60],b[start:start+60]
        body,*_,s=struct.unpack_from('<6I',aa)
        assert s==i+1
        if s in change:
            assert aa[:24]==bb[:24] and aa[52:]==bb[52:];owners[body]=s
        else: assert aa==bb
    end=60+p[1][7]*60;assert a[end:]==b[end:] and len(owners)==4
    assert p[1][-1]==q[1][-1]==bytes.fromhex(sha(old)) and r[1][-1]==bytes.fromhex(sha(new))
    assert p[1][1:6]==q[1][1:6]==r[1][1:6] and p[1][-3:-1]==q[1][-3:-1]==r[1][-3:-1]
    fields=('muscle','endpoint','body_index','source_site_index','route_node_index','source_local_point_m','resolved_local_point_m','endpoint_migration_m')
    newrows=[];newenv=[];emeta=[]
    for i,(x,y,z) in enumerate(zip(p[0]['endpoints'],q[0]['endpoints'],r[0]['endpoints'])):
        assert all(x[k]==y[k]==z[k] for k in fields)
        if x['body_index'] in owners:
            # Require old re-emission to reproduce the actual original law,
            # including its complete force distribution. No fallback migration.
            assert p[2][i]==q[2][i] and x==y
            assert not x['surface'] or z['surface']
            origin=r;controls.append(i)
            if z['surface']:assert z['surface']['bone_stable_id']==owners[z['body_index']]
        else:
            assert q[2][i]==r[2][i] and y==z
            if x['surface']:
                assert x['surface']['bone_stable_id'] not in change
                sem=x['surface'].get('semantic_enthesis_map',{})
                assert not change.intersection(sem.get('node_bone_stable_ids',[]))
            origin=p
        ep,env=origin[2][i];row=bytearray(ep)
        if env:
            struct.pack_into('<I',row,24,len(newenv));newenv.append(env)
        newrows.append(row);emeta.append(origin[0]['endpoints'][i])
    dst=list(r[1]);dst[6]=len(newenv)
    out.mkdir(exist_ok=True);output=out/TNAME
    output.write_bytes(struct.pack('<8s10I32s32s32s',*dst)+b''.join(newrows)+b''.join(newenv))
    d=p[0];before=d['coverage'].copy();newcov=d['coverage']
    for key,val in list(newcov.items()):
        if key.endswith('_count'):newcov[key]=val+r[0]['coverage'][key]-q[0]['coverage'][key]
    surfaces=[v['surface'] for v in emeta if v['surface']]
    assert newcov['distributed_surface_envelope_count']==len(surfaces)==len(newenv)
    newcov.update(surface_coverage_fraction=len(surfaces)/len(emeta),
        maximum_admitted_surface_distance_m=max(v['surface_distance_m'] for v in surfaces),
        maximum_admitted_sampled_total_force_amplification=max(v['sampled_total_force_amplification'] for v in surfaces),
        maximum_endpoint_migration_m=max(v['endpoint_migration_m'] for v in emeta))
    from collections import Counter
    d['admission']['rejection_counts']=dict(Counter(v['admission_reason'] for v in emeta if not v['surface']))
    d['payload'].update(sha256=sha(output),bytes=output.stat().st_size)
    d['source']['bodyparts3d_bone_payload']=r[0]['source']['bodyparts3d_bone_payload']
    d['source']['regional_recook']=dict(parent_tendon_sha256=sha(td),parent_bone_sha256=sha(old),compiled_bone_sha256=sha(new),
        changed_bone_ids=sorted(change),recooked_endpoint_indices=controls,unchanged_endpoint_laws=len(emeta)-len(controls),
        baseline_regional_compiler_sha256=sha(regional0/TNAME),candidate_regional_compiler_sha256=sha(regional1/TNAME))
    d['endpoints']=emeta;output.with_suffix('.manifest.json').write_text(json.dumps(d,separators=(',',':'),allow_nan=False)+'\n')
    verify=unpack(output)
    for i,v in enumerate(emeta):
        assert verify[2][i]==(r if v['body_index'] in owners else p)[2][i]
    compare(td.with_suffix('.manifest.json'),output.with_suffix('.manifest.json'),out/'comparison.json')
    print(len(controls),len(emeta),len(newenv),sha(output))
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
