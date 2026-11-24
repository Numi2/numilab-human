"""Carry registered rib attachment vertex IDs across byte-identical source records."""
from pathlib import Path
import struct,sys,json,re,hashlib,os
import numpy as np
from rebind_ligament_material import banks
from compile_connected_joint import digest
from run_cavity_motion import execute_native
def surface_record(path,stable):
    memory=path.read_bytes();magic,abi,n,v,i,_,_=struct.unpack_from('<8s5I32s',memory)
    if abi!=5 or 60+32*n+24*v+4*i!=len(memory):raise ValueError('Unexpected anatomy layout')
    rec=np.frombuffer(memory,'<u4',8*n,60).reshape(-1,8)
    rows=rec[rec[:,5]==stable]
    if len(rows)!=1:raise ValueError('Muscle lacks a unique typed identifier')
    b,fv,nv,fi,ni,s,l,extra=map(int,rows[0]);start=60+32*n+24*fv
    faces=np.frombuffer(memory,'<u4',ni,60+32*n+24*v+4*fi)
    if ni%3 or faces.min()<fv or faces.max()>=fv+nv:raise ValueError('Invalid local triangles')
    return np.asarray([b,s,l,extra,nv,ni],'<u4').tobytes()+memory[start:start+24*nv]+(faces-fv).astype('<u4').tobytes()
def run(job,base,old_bone,out):
    read=lambda p:json.loads(p.read_bytes());parent=base/'launch.json';cfg=read(parent);args=cfg['argv']
    receipt=Path(args[args.index('--resting-anatomy-receipt')+1]);active=Path(args[args.index('--torso-anatomy-payload')+1]);bone=Path(args[3])
    for p in (receipt,active,bone):
        if cfg['pins'].get(str(p))!=digest(p):raise ValueError('Launch identity mismatch')
    original=Path(read(job/'registered-organ-viewer/records.json')['source_path'])
    source=read(receipt);binding=source['functional_bindings']
    old_a,old_b,new_a,new_b=map(digest,[original,old_bone,active,bone])
    if binding['anatomy_payload_sha256']!=new_a or binding['bones_payload_sha256']!=new_b:raise ValueError('Parent channels are for another asset')
    before=banks(old_bone);after=banks(bone);proof=[];muscles={}
    for key,row in binding.items():
        m=re.fullmatch(r'intercostal_(\d+)_rib(\d+)_motion_attachment',key)
        if not m:continue
        muscle,rib=map(int,m.groups());surface=row['surface'];channel=row['existing_rib_channel'];ident=channel['body_index'],rib
        if muscle!=surface['stable_id'] or rib!=channel['stable_id'] or row['anatomy_payload_sha256']!=old_a or row['bones_payload_sha256']!=old_b:raise ValueError('Unexpected source identities')
        if muscle not in muscles:
            sa=surface_record(original,muscle);sb=surface_record(active,muscle)
            if sa!=sb:raise ValueError('The intercostal surface has changed')
            muscles[muscle]=hashlib.sha256(sa).hexdigest()
        left=before[ident];right=after[ident]
        if any(x.shape!=y.shape or x.tobytes()!=y.tobytes() for x,y in zip(left,right)):raise ValueError('Bone geometry or its body-local registration has changed')
        ids=np.array(surface['interface_local_vertex_ids'],dtype='<u4');vertices=hashlib.sha256(ids.tobytes()).hexdigest()
        if not len(ids) or (ids[1:]<=ids[:-1]).any() or int(ids.max())>=surface['local_vertex_count'] or vertices!=surface['interface_local_vertex_ids_sha256']:raise ValueError('Invalid attached vertex inventory')
        proof.append(dict(binding=key,body=ident[0],rib=rib,paired_vertices=len(ids),vertex_hash=vertices,intercostal_hash=muscles[muscle],typed_rib_hash=hashlib.sha256(b''.join(z.tobytes() for z in left)).hexdigest()))
        row.update(anatomy_payload_sha256=new_a,bones_payload_sha256=new_b)
    if len(proof)!=3:raise ValueError('Unexpected attachment inventory')
    out.mkdir();p=out/'resting-anatomy-receipt.json';p.write_text(json.dumps(source,sort_keys=True,indent=2)+'\n')
    (out/'intercostal-registration.json').write_text(json.dumps(dict(parent=digest(receipt),result=digest(p),original_anatomy=old_a,current_anatomy=new_a,original_bones=old_b,current_bones=new_b,bindings=proof),indent=2)+'\n')
    args[args.index('--resting-anatomy-receipt')+1]=str(p);args[4]=str(out/'native-traces');cfg['parent_launch']=digest(parent);cfg['pins'][str(p)]=digest(p)
    env={k:v.replace(str(base)+'/',str(out)+'/') for k,v in cfg['environment'].items()};cfg['environment']=env
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(execute_native(args,cfg['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|env,out,minimum_free_bytes=4*2**30,timeout_s=400),flush=True)
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
