#!/usr/bin/env python3
"""Verify the 635 closed-fat compaction and typed source map without altering geometry."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def validate_arrays(points, faces, data, ip, iff, dp, dff, rim, facets):
    require(points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all(), 'invalid points')
    require(faces.ndim == 2 and faces.shape[1] == 3 and np.issubdtype(faces.dtype, np.integer), 'invalid faces')
    used=np.asarray(data['output_vertex_input_rows'])
    vr=np.asarray(data['output_vertex_source_layer'])
    vi=np.asarray(data['output_vertex_source_row'])
    fr=np.asarray(data['face_source_layer'])
    fi=np.asarray(data['face_source_face_row'])
    flip=np.asarray(data['face_source_winding_reversed'])
    require(used.shape == vr.shape == vi.shape == (len(points),), 'vertex map shape')
    require(fr.shape == fi.shape == flip.shape == (len(faces),), 'face map shape')
    require(np.issubdtype(used.dtype,np.integer) and np.issubdtype(vi.dtype,np.integer) and np.issubdtype(fi.dtype,np.integer), 'source rows must be integers')
    require(np.all((vr == 2)|(vr == 3)), 'unknown vertex source role')
    require(np.all((fr == 2)|(fr == 3)|(fr == 4)), 'unknown face source role')
    require(np.all(used >= 0) and np.all(used < len(ip)+len(dp)), 'input row out of range')
    require(np.all(used[1:] > used[:-1]), 'compaction rows must be strictly ordered')
    require(np.array_equal(used, np.where(vr == 2, vi, vi+len(ip))), 'typed source/input row disagreement')
    require(np.all(vi >= 0) and np.all(vi < np.where(vr==2,len(ip),len(dp))), 'typed vertex row out of range')
    rebuilt=np.empty_like(points)
    for role,source in ((2,ip),(3,dp)):
        mask=vr==role
        rebuilt[mask]=source[vi[mask]]
    require(np.array_equal(rebuilt,points), 'source point reconstruction mismatch')
    require(np.all(faces >= 0) and np.all(faces < len(points)), 'face index out of range')
    inverse=np.full(len(ip)+len(dp),-1,np.int64)
    inverse[used]=np.arange(len(used))
    raw=np.empty_like(faces,dtype=np.int64)
    for role,source,offset,flipped in ((2,iff,0,False),(3,dff,len(ip),True)):
        mask=fr==role
        require(np.all(fi[mask]>=0) and np.all(fi[mask]<len(source)), 'typed face row out of range')
        require(np.all(flip[mask]==flipped), 'source winding flag mismatch')
        raw[mask]=source[fi[mask]][:,[0,2,1] if flipped else [0,1,2]]+offset
    ids=np.asarray(rim['fat_headwall_source_face_rows'])
    require(len(set(map(int,ids)))==len(ids), 'duplicate rim source identity')
    lookup={int(value):i for i,value in enumerate(ids)}
    for row in np.flatnonzero(fr==4):
        facet_row=int(fi[row])
        require(0<=facet_row<len(facets), 'rim CSV row out of range')
        facet=facets[facet_row]
        require(facet['source_partition']=='fat' and facet['face_source_role']=='donor_fat_ID_headwall', 'wrong rim facet identity')
        source_id=int(facet['face_source_row'])
        require(source_id in lookup, 'unknown rim source identity')
        idx=lookup[source_id]
        require(np.array_equal(json.loads(facet['parent_source_node_rows']),rim['fat_headwall_parent330_node_rows'][idx]), 'rim parent node identity mismatch')
        require(not bool(flip[row]), 'unexpected rim reversal')
        raw[row]=rim['fine_fat_rim_faces'][idx]
    require(np.all(raw>=0) and np.all(raw<len(inverse)), 'raw source face row out of range')
    require(np.all(inverse[raw]>=0) and np.array_equal(inverse[raw],faces), 'source face reconstruction mismatch')
    require(np.array_equal(np.unique(faces),np.arange(len(points))), 'unreferenced compacted point')
    return {'points':len(points),'faces':len(faces),'typed_point_reconstruction':'PASS','typed_face_reconstruction':'PASS'}

def run(geometry, source_map, registry_path, rim_path, check_negative_controls=False):
    registry=json.loads(Path(registry_path).read_text())
    for role in ('2','3','4'):
        require(sha(registry[role]['path'])==registry[role]['sha256'], 'source hash mismatch')
    with np.load(geometry,allow_pickle=False) as z:
        points=z['points_world_m_f64'];faces=z['faces']
    with np.load(source_map,allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    with np.load(registry['2']['path'],allow_pickle=False) as z:
        ip=z['points_world_m'];iff=z['faces']
    with np.load(registry['3']['path'],allow_pickle=False) as z:
        dp=z['points_world_m'];dff=z['faces']
    with np.load(rim_path,allow_pickle=False) as z:
        rim={k:z[k] for k in z.files}
    with Path(registry['4']['path']).open(newline='') as f:
        facets=list(csv.DictReader(f))
    args=(points,faces,data,ip,iff,dp,dff,rim,facets)
    result=validate_arrays(*args)
    controls=[]
    if check_negative_controls:
        changes=[
            ('wrong_vertex_source_role','output_vertex_source_layer',0,4),
            ('negative_vertex_source_row','output_vertex_source_row',0,-1),
            ('wrong_compaction_row','output_vertex_input_rows',0,len(ip)+len(dp)),
            ('wrong_face_source_role','face_source_layer',0,1),
            ('negative_face_source_row','face_source_face_row',0,-1),
            ('wrong_source_winding','face_source_winding_reversed',0,True),
        ]
        for label,key,row,value in changes:
            bad=dict(data);bad[key]=data[key].copy();bad[key][row]=value
            try:
                validate_arrays(points,faces,bad,ip,iff,dp,dff,rim,facets)
            except ValueError as error:
                controls.append({'control':label,'status':'REJECTED','reason':str(error)})
            else:
                raise ValueError('negative control accepted: '+label)
        altered=points.copy();altered[0,0]=np.nextafter(altered[0,0],np.inf)
        try:validate_arrays(altered,faces,data,ip,iff,dp,dff,rim,facets)
        except ValueError as error:controls.append({'control':'changed_coordinate','status':'REJECTED','reason':str(error)})
        else:raise ValueError('changed coordinate accepted')
        altered=faces.copy();altered[0,[0,1]]=altered[0,[1,0]]
        try:validate_arrays(points,altered,data,ip,iff,dp,dff,rim,facets)
        except ValueError as error:controls.append({'control':'changed_face_winding','status':'REJECTED','reason':str(error)})
        else:raise ValueError('changed face winding accepted')
    return {'status':'PASS_TYPED_SOURCE_RECONSTRUCTION','geometry_sha256':sha(geometry),'source_map_sha256':sha(source_map),'source_registry_sha256':sha(registry_path),'rim_source_map_sha256':sha(rim_path),**result,'negative_controls':controls,'physical_admission':False,'runtime':0,'accepted_steps':0}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--source-map',type=Path,required=True)
    p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--rim-map',type=Path,required=True)
    p.add_argument('--negative-controls',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=run(a.geometry,a.source_map,a.registry,a.rim_map,a.negative_controls)
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
