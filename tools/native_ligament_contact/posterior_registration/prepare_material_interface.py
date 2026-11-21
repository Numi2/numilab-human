"""Partition the existing registered muscle material against actual cortical and deep-fat boundaries."""
import json,sys
from pathlib import Path
import numpy as np
import manifold3d as m
from numilab_human.capsular_material_registration import read_material_banks,material_bank,emitted_body_vertices
from material_context import applied,sha,get_layers
from prepare_posterior_muscles import closed,mesh_volume
def close_remote_boundary(x,f,roi):
    # Cap only the oriented source holes outside the pelvic analysis domain.
    # These temporary triangles may never contribute a muscle output face.
    edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    _,idx,count=np.unique(np.sort(edges,axis=1),axis=0,return_index=True,return_counts=True)
    if (count>2).any():raise ValueError('Ambiguous deep material boundary')
    b=edges[idx[count==1]];after=dict(map(tuple,b.tolist()))
    if len(after)!=len(b) or len(set(after.values()))!=len(b):raise ValueError('Branched source opening')
    verts=list(x);faces=list(f);caps=[]
    while after:
        j=min(after);loop=[j];k=after.pop(j)
        while k!=j:loop.append(k);k=after.pop(k)
        if len(loop)<3 or x[loop,1].min()<roi[1]+.05:raise ValueError('Opening approaches posterior material')
        center=np.mean(x[loop],axis=0);t=len(verts);verts.append(center)
        faces.extend([loop[(i+1)%len(loop)],a,t] for i,a in enumerate(loop))
        caps.append(loop)
    return np.array(verts),np.array(faces),caps

def solid(x,f,identity):
    v,iv=np.unique(x,axis=0,return_inverse=True);t=iv[f]
    vs=mesh_volume(v,t)
    if vs<0:t=t[:,[0,2,1]]
    ids=m.Manifold.reserve_ids(1)
    source=m.Mesh64(v.astype(float),np.ascontiguousarray(t,dtype='uint64'),run_index=np.array([0,3*len(t)],dtype='uint64'),run_original_id=np.array([ids],dtype='uint32'),face_id=np.arange(len(t),dtype='uint64'))
    obj=m.Manifold(source)
    return obj,ids,dict(name=identity,status=str(obj.status()),source_vertices=len(x),welded_vertices=len(v),source_faces=len(t),signed_volume_m3=vs,mesh=closed(v,t))
def read_inputs(config,receipt,capture,base):
    args=json.loads(config.read_text())['argv'];bone=Path(args[3]);bank=read_material_banks(bone)
    rc=json.loads(receipt.read_text());frames={r['body_index']:np.array(r['position_m']+r['quaternion_xyzw']) for r in rc['initial_anatomical_registration']['body_poses']}
    data={}
    for sid in (1,2,3,18,19):
        bid=bank[2][sid][0];a,row,v,f=material_bank(bank,sid,bid)
        data['bone_'+str(sid)]=(applied(frames[bid],emitted_body_vertices(v,row)).astype('f4'),f)
    for sid in (29,30):
        d=np.load(capture/('entrance-%d.npz'%sid));data['muscle_'+str(sid)]=(d['xyz'].astype('f4'),d['faces'])
    rec=base/'qualified/motion-008/step-8.receipt.json';pack=base/'qualified/native-coupled-006/native/resting-human.mrvpack'
    for role,(v,f) in get_layers(pack,rec).items():data['skin_'+role]=(v,f)
    return data,dict(bones=bank[1],native_entrance=rc['base_pack_file_sha256'],layer_parent_sha256=sha(pack),current_receipt_sha256=sha(receipt))
def run(config,receipt,capture,base,out):
    out.mkdir(exist_ok=True);surfs,lineage=read_inputs(config,receipt,capture,base)
    ids={};objs={};meta={}
    for name,(v,f) in surfs.items():
        original_count=len(f);caps=[]
        if name=='skin_deep':
            roi=np.maximum(surfs['muscle_29'][0].max(0),surfs['muscle_30'][0].max(0))
            v,f,caps=close_remote_boundary(v,f,roi)
        obj,identity,row=solid(v,f,name);row.update(original_triangle_count=original_count,temporary_analysis_loops=caps);objs[name]=obj;ids[identity]=name;meta[name]=row
        print(name,row['status'],row['mesh'],flush=True)
    (out/'source.json').write_text(json.dumps(dict(lineage=lineage,surfaces=meta),indent=2)+'\n')
    def check(a):
        if a.status()!=m.Error.NoError or a.is_empty():raise ValueError(str(a.status()))
    for name in objs:
        if name.startswith(('bone','muscle')) or name=='skin_deep':check(objs[name])
    for sid,bones in [(29,(1,2,18)),(30,(1,3,19))]:
        key='muscle_'+str(sid);initial=objs[key];conform=initial
        row=dict(original_ml=initial.volume()*1e6,removed_ml={})
        for k in ['bone_'+str(b) for b in bones]:
            previous=conform;conform=conform-objs[k];check(conform);row['removed_ml'][k]=(previous.volume()-conform.volume())*1e6
        previous=conform;conform=conform^objs['skin_deep'];check(conform)
        row['removed_ml']['skin_deep_exterior']=(previous.volume()-conform.volume())*1e6
        native=conform.to_mesh64();v=native.vert_properties[:,:3];f=np.asarray(native.tri_verts,int)
        run=np.repeat(native.run_original_id,np.diff(native.run_index)//3)
        names=np.array([ids[int(i)] for i in run]);parent=np.array(native.face_id)
        if np.any((names=='skin_deep')&(parent>=meta['skin_deep']['original_triangle_count'])):raise ValueError('Remote cap entered physical domain')
        row.update(conforming_volume_ml=conform.volume()*1e6,parts=len(conform.decompose()),emitted_topology=closed(v.astype('f4'),f),parent_faces={k:int(sum(names==k)) for k in np.unique(names)})
        np.savez_compressed(out/(key+'.npz'),vertices=v,triangles=f,parent_bank=names,parent_face=parent)
        (out/(key+'.json')).write_text(json.dumps(row,indent=2)+'\n');print(row,flush=True)
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
