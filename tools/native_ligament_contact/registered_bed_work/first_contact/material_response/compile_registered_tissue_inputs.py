"""Compile the registered shoulder frames through native muscle, passive-carrier and skin bindings."""
import json,os,struct,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from compile_connected_joint import matrix,digest


def transform(raw,offset,R,t):
    b,*a=struct.unpack_from('<I8f',raw,offset)
    prior=Rotation.from_quat(a[3:7]);rot=Rotation.from_matrix(R)*prior
    struct.pack_into('<7f',raw,offset+4,*(R@a[:3]+t),*rot.as_quat())
    new=struct.unpack_from('<I8f',raw,offset)
    err=float(np.max(np.abs(R@prior.as_matrix()*a[7]-Rotation.from_quat(new[4:8]).as_matrix()*new[8])))
    if err>3e-6 or np.linalg.norm(new[1:4]-(R@a[:3]+t))>2e-6:raise ValueError('Float32 binding lost registration')
    return err


def run(job,rigid,bones,vessels,out):
    argv=json.loads((job/'launch.json').read_bytes())['argv']
    poses=json.loads((job/'native-initial-bodies.json').read_bytes())['all_bodies']
    bpath=next(bones.glob('*.nhbones'));bm=json.loads(bpath.with_suffix('.manifest.json').read_bytes())
    orig=Path(argv[3]);old_fp=struct.unpack_from('<I',orig.read_bytes(),24)[0]
    new_fp=int(bm['payload']['registration_fingerprint32'],16)
    reg=json.loads((bones/bm['source']['registration']['file']).read_bytes())['regional_derivation']['connected_shoulder_registration']
    rb=rigid/'source-bound.nhrigid'
    if digest(rb)!=reg['rigid_sha256'] or old_fp==new_fp:raise ValueError('Wrong registration')
    shift={v['body']:np.asarray(v['body_local_transform']) for v in reg['changed'].values()}
    out.mkdir();reports={};copied={}
    def tissue(src,dest):
        raw=bytearray(src.read_bytes());h=struct.unpack_from('<8s6I32s',raw)
        if h[0]!=b'NHTISS4\0' or h[1]!=5 or h[6]!=old_fp or 64+32*h[2]+36*h[3]+56*h[4]+4*h[5]!=len(raw):
            raise ValueError('Unmatched passive or muscle surface')
        sha=digest(src)
        if sha in copied:os.link(copied[sha],dest);return
        rows=[];errors=[]
        for i in range(h[3]):
            off=64+32*h[2]+36*i;b=struct.unpack_from('<I',raw,off)[0]
            if b in shift:
                A=shift[b];errors.append(transform(raw,off,A[:3,:3],A[:3,3]));rows.append(i)
        struct.pack_into('<I',raw,28,new_fp)
        dest.write_bytes(raw);copied[sha]=dest
        reports[dest.name]=dict(source=sha,result=digest(dest),rows=h[2],bindings=h[3],cortical_bindings_changed=len(rows),max_linear_map_error=max(errors,default=0.))
    receipt=vessels/'resting-anatomy-receipt.json';d=json.loads(receipt.read_bytes());bd=d['passive_tissue_parent_bindings']
    child=Path(argv[argv.index('--soft-tissue-payload')+1]);soft=out/child.name
    if digest(child)!=bd['child_payload_sha256']:raise ValueError('Unmatched passive child')
    tissue(child,soft);bd['child_payload_sha256']=digest(soft)
    for i,s in enumerate(bd['source_payloads']):
        orig_parent=Path(s['path']);dest=out/f'parent-{i}.nhtissue'
        if digest(orig_parent)!=s['sha256']:raise ValueError('Unmatched passive parent')
        tissue(orig_parent,dest);s.update(path=str(dest.resolve()),sha256=digest(dest))
    # Single-owner NHA surfaces on girdle bones carry the same local correction.
    # The two blended vascular children were already registered on body 91.
    nha=next(vessels.glob('*.nhanatomy'));raw=bytearray(nha.read_bytes())
    h=struct.unpack_from('<8s5I32s',raw);vo=60+32*h[2];nrows=0
    if h[0]!=b'NHANAT1\0' or h[1]!=5 or h[5]!=old_fp or digest(nha)!=bd['anatomy_payload_sha256']:raise ValueError('Unmatched torso anatomy')
    for i in range(h[2]):
        b,start,n,_,_,_,_,_=struct.unpack_from('<8I',raw,60+32*i)
        if b in shift:
            A=shift[b];v=np.frombuffer(raw,'<f4',6*n,vo+24*start).reshape(-1,6)
            v[:,:3]=v[:,:3]@A[:3,:3].T+A[:3,3];v[:,3:]=v[:,3:]@A[:3,:3].T;nrows+=1
    struct.pack_into('<I',raw,24,new_fp);anh=out/nha.name;anh.write_bytes(raw);nhash=digest(anh)
    d['payload'].update(path=str(anh.resolve()),sha256=nhash,input_payload_sha256=digest(nha),registration_fingerprint32=f'{new_fp:08x}')
    for k in (bd,d['functional_bindings']):k['anatomy_payload_sha256']=nhash
    reports[anh.name]=dict(source=digest(nha),result=nhash,single_owner_rows_transported=nrows)
    # Preserve the actual supported outer skin while retaining every source
    # weight. W_new B_new = W_old B_old independently for each affected donor.
    skin=Path(argv[argv.index('--skin-payload')+1]);data=bytearray(skin.read_bytes());hs=struct.unpack_from('<8s5I32s',data)
    if hs[0]!=b'NHSKIN1\0' or hs[1]!=5 or hs[5]!=old_fp or len(data)!=60+36*hs[2]+56*hs[3]+4*hs[4]+4*hs[3]*hs[2]:raise ValueError('Unexpected full skin ABI')
    rawrigid=rb.read_bytes()
    def world(b):
        p=poses[b]['current'];return matrix(p['position_world_m']+p['orientation_world_xyzw'])
    base={}
    for b,f in zip((84,34),json.loads((rigid/'frames.json').read_bytes())):
        W=world(b);base[b]=W@matrix(f['parent_translation_base_m']+f['parent_rotation_xyzw'])@np.linalg.inv(W)
    vertices=np.ndarray((hs[3],3),'<f4',buffer=data,offset=60+36*hs[2],strides=(56,4))
    weight=np.frombuffer(data,'<f4',hs[3]*hs[2],len(data)-4*hs[2]*hs[3]).reshape(hs[3],hs[2])
    err=0.;changed=0;influences=0
    for i in range(hs[2]):
        offset=60+36*i;b,*a=struct.unpack_from('<I8f',data,offset);child=b
        while child and child-1 not in base:
            parent,k=struct.unpack_from('<2I',rawrigid,224+160*157+144*(child-1))
            if k!=child or parent>=k:raise ValueError('Invalid physical arm subtree')
            child=parent
        if not child:continue
        old=world(b);new=base[child-1]@old;delta=np.linalg.solve(new,old)
        transform(data,offset,delta[:3,:3],delta[:3,3]);_,*q=struct.unpack_from('<I8f',data,offset)
        v=vertices[weight[:,i]>0].astype(float)
        def apply(W,k):
            local=Rotation.from_quat(k[3:7]).apply(v*k[7])+k[:3]
            return local@W[:3,:3].T+W[:3,3]
        es=np.linalg.norm(apply(old,a)-apply(new,q),axis=1)
        err=max(err,float(es.max(initial=0.)));changed+=1;influences+=len(es)
    if err>2e-6:raise ValueError('Protected skin altered at release')
    struct.pack_into('<I',data,24,new_fp);sk=out/skin.name;sk.write_bytes(data)
    reports[sk.name]=dict(source=digest(skin),result=digest(sk),vertices=hs[3],changed_bindings=changed,checked_positive_weights=influences,max_supported_material_point_error_m=err)
    d['provenance']['connected_shoulder_tissue_registration']=dict(original_receipt=digest(receipt),rigid=digest(rb),bones=digest(bpath),source_owners=reports)
    (out/receipt.name).write_text(json.dumps(d,separators=(',',':'),allow_nan=False)+'\n')
    (out/'result.json').write_text(json.dumps(dict(rigid=digest(rb),bones=digest(bpath),source_owners=reports),indent=2)+'\n')
    print('skin error',err,'girdle anatomy rows',nrows,'muscle bindings',reports[soft.name]['cortical_bindings_changed'])


if __name__=='__main__':run(*(Path(k).resolve() for k in sys.argv[1:]))
