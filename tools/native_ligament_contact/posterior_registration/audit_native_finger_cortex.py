"""Measure actual fourth/fifth-digit cortical crossings in a native MRVPACK2 capture."""
import hashlib,json,mmap,os,re,sys,struct
from pathlib import Path
import numpy as np
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(w,launch,pack,out):
    sys.path[:0]=[str(w/'resting-surface-clearance-001/src'),str(w/'resting-surface-clearance-001'),str(w/'resting-clearance-and-atrium-005')]
    import audit
    from fast_skin_intersection_audit import install_for_float32
    install_for_float32()
    cfg=json.loads(launch.read_text());receipt=json.loads(pack.with_suffix('.receipt.json').read_text())
    bone=Path(next(a for a in cfg['argv'] if a.endswith('.nhbones')));m=json.loads(bone.with_suffix('.manifest.json').read_text())
    if cfg['pins'][str(bone)]!=m['payload']['sha256'] or sha(bone)!=m['payload']['sha256'] or receipt['pack_file_sha256']!=sha(pack):raise ValueError('Bone or pose pin mismatch')
    raw=bone.read_bytes()
    hdr=struct.Struct('<8s5I32s');record=struct.Struct('<6I8fI')
    magic,abi,nb,nv,ni,_,_=hdr.unpack_from(raw)
    if (magic!=b'NHBONES1' or abi!=3 or nb!=len(m['source']['anchors']) or
        len(raw)!=hdr.size+record.size*nb+nv*24+ni*4):raise ValueError('Invalid native cortical layout')
    elem=np.frombuffer(raw,'<u4',count=ni,offset=hdr.size+record.size*nb+nv*24)
    faces_by_stable={}
    poses={r['body_index']:r for r in receipt['accepted_body_poses']};mapping={}
    for i,a in enumerate(m['source']['anchors']):
        name=a['myosim_body']
        if re.fullmatch(r'([45]proxph|midph[45]|distph[45])_[lr]',name):
            b,v0,vc,f0,ic,stable,*tail=record.unpack_from(raw,hdr.size+i*record.size)
            if stable!=i+1 or b!=a['core_body_index'] or tail[-1]!=a['source_record_index'] or ic%3:
                raise ValueError('Emitted cortical payload differs from its anatomical identity')
            face=elem[f0:f0+ic].reshape(-1,3)
            faces_by_stable[stable]=np.searchsorted(np.unique(face),face)
            mapping[stable]=(name,b,ic//3,a['member_id'])
    if len(mapping)!=12:raise ValueError('Expected twelve named phalanges')
    banks={};saved={}
    with pack.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as buf:
        for (semantic,stable),prim,v,f in audit.primitives(buf,poses):
            if semantic!=51004 or stable not in mapping:continue
            name,body,n,member=mapping[stable]
            if n!=len(f) or name in banks or not np.array_equal(f,faces_by_stable[stable]):raise ValueError('Correspondence with cortical source failed')
            exact=audit.clearance._exact_surface_records(v,f);index=audit.intersections._prepare_surface_aabb(exact)
            banks[name]=(exact,index)
            saved[name+'_v']=v;saved[name+'_f']=f
    if len(banks)!=12:raise ValueError('Native capture lacks required phalanges')
    rows=[]
    for side in ['r','l']:
        for a in [x+'_'+side for x in ['4proxph','midph4','distph4']]:
            for b in [x+'_'+side for x in ['5proxph','midph5','distph5']]:
                rec=audit.intersections._audit_pair_prepared_first(banks[a][1],banks[b][0])
                rows.append(dict(first=a,second=b,count=rec['count'],triangle_pairs=rec['triangle_pairs']))
    report={'step':receipt['accepted_step'],'pack_sha256':sha(pack),'cortical_payload_sha256':sha(bone),'capture_sha256':receipt['captured_vertex_buffer_sha256'],
        'member_by_stable_id':mapping,'tested_pairs':len(rows),'total_triangle_intersections':sum(row['count'] for row in rows),'pairs':rows,
        'scope':'Actual cortical surfaces of fourth and fifth proximal, middle, and distal phalanges, at this instant. Excludes other fingers, soft tissues and between-frame impacts.'}
    out.write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(out.with_suffix('.npz'),**saved)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pack',type=Path);parser.add_argument('out',type=Path)
    parser.add_argument('--workspace',type=Path,default=os.environ.get('W'))
    parser.add_argument('--launch',type=Path,default=os.environ.get('FINGER_CAPTURE_CONFIG'))
    args=parser.parse_args()
    if args.workspace is None or args.launch is None:parser.error('Provide the source workspace and the native launch configuration')
    run(args.workspace,args.launch,args.pack,args.out)
