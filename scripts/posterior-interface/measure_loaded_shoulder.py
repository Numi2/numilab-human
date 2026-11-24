"""Measure the registered tissues actually crossed by the dominant native bed impulse."""
import csv
import json
import mmap
from pathlib import Path
import runpy
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from compile_connected_joint import digest
from trace_loaded_columns import hits
from pose_glutes import parts,model


def run(case,volume,common,registry,out):
    receipt=next((case/'native-traces/accepted-geometry').glob('*.receipt.json'))
    frame=json.loads(receipt.read_bytes())
    assert frame['physical_endpoint']=='accepted'
    poses={p['body_index']:p for p in frame['accepted_body_poses']}
    rigid={k:np.r_[p['position_m'],p['quaternion_xyzw']] for k,p in poses.items()}
    cfg=json.loads((case/'launch.json').read_bytes());args=cfg['argv']
    tissue=Path(args[args.index('--soft-tissue-payload')+1])
    pack=Path(frame['accepted_pack_path']); assert digest(pack)==frame['pack_file_sha256']
    history=case/'physical-posterior-bed-history.csv'
    with history.open() as stream:
        impulse={int(row['registered_skin_node']):float(row['terminal_microstep_normal_impulse_Ns']) for row in csv.DictReader(stream)}
    normal=np.array(json.loads(common.read_bytes())['bed_normal'])
    horizontal=np.cross([0.,1.,0.],normal);horizontal/=np.linalg.norm(horizontal)
    axes=np.column_stack((horizontal,np.cross(normal,horizontal),normal))
    receiver=max(impulse,key=impulse.get)
    skin=np.loadtxt(case/'physical-accepted.node',skiprows=1,usecols=(1,2,3))
    walls=np.load(volume/'material-walls.npz')
    ring=np.unique(walls['O'][np.any(walls['O']==receiver,axis=1)])
    ids=np.unique(np.r_[ring,list(impulse)])
    queries=skin[ids]@axes
    rows={int(i):dict(node=int(i),terminal_impulse_Ns=impulse.get(i,0.),adjacent=bool(i in ring),muscle=[],bone=[],walls={})for i in ids}
    for k in 'IDC':
        for node,spans in zip(ids,hits(skin@axes,walls[k],k,queries)):
            if spans:rows[node]['walls'][k]=spans[0][0]
    api=runpy.run_path(str(registry));sources={}
    with pack.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as buf:
        for key,_,pts,tri in api['primitives'](buf,poses):
            if key[0] not in (51005,51004):continue
            if key==(51005,83):
                raw=parts(tissue,83);v,_,_,_=model(raw,rigid,1)
                s=raw[0];source_ids=np.unique(raw[3]);s=s[source_ids];v=v[source_ids]
                f=np.searchsorted(source_ids,raw[3])
                if len(v)!=len(pts) or not np.array_equal(f,tri):
                    raise ValueError('Source face ancestry differs')
                def signed_volume(x):
                    t=(x-x.mean(0))[f]
                    return np.sum(np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])))/6.
                sources=dict(source_payload=digest(tissue),donors=raw[2],surface_faces=len(f),max_source_to_render_m=float(np.linalg.norm(v-pts,axis=1).max()),
                             atlas_volume_m3=float(signed_volume(s)),native_display_volume_m3=float(signed_volume(pts)))
            xy=pts@axes
            if np.any(xy[:,:2].max(0)<queries[:,:2].min(0)) or np.any(xy[:,:2].min(0)>queries[:,:2].max(0)):continue
            name='bone' if key[0]==51004 else 'muscle'
            for node,spans in zip(ids,hits(xy,tri,key[1],queries)):
                rows[node][name].extend((h,int(k))for h,k in spans if h<.18)
    for row in rows.values():
        for name in ('muscle','bone'):row[name].sort()
    if not sources:raise ValueError('Missing right infraspinatus')
    result=dict(accepted_receipt=digest(receipt),history=digest(history),skin=digest(case/'physical-accepted.node'),
                surface_pack=digest(pack),operator_registry=digest(registry),dominant_skin_node=int(receiver),
                fraction=impulse[receiver]/sum(impulse.values()),normal=normal.tolist(),right_infraspinatus=sources,
                sites=list(rows.values()))
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('dominant',receiver,result['fraction'],'native source',sources,flush=True)

if __name__=='__main__':
    run(*(Path(s).resolve() for s in sys.argv[1:]))
