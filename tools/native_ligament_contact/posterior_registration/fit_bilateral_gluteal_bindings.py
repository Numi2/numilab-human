"""Test coupled gluteal source registration at the retained NHTISSUE entrance."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from bilateral_bindings import decode
from prepare_posterior_muscles import load_registered,interface_audit,mesh_volume
from compare_hand_interfaces import query
from material_context import sha
def pair_count(exe,v0,f0,v1,f1):
    f=np.vstack([f0,f1+len(v0)])
    hits=query(exe,np.vstack([v0,v1]),f,np.arange(len(f))<len(f0))
    return sum(i<len(f0)<=j for i,j in hits)
def harmonic(v,f,seeds,values):
    edge=np.unique(np.sort(np.vstack((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1),axis=0)
    i,j=edge.T;coeff=1/np.maximum(np.linalg.norm(v[i]-v[j],axis=1),1e-12)
    lap=coo_matrix((np.r_[coeff,coeff,-coeff,-coeff],(np.r_[i,j,i,j],np.r_[i,j,j,i])),shape=(len(v),len(v))).tocsr()
    free=np.setdiff1d(np.arange(len(v)),seeds);w=np.zeros(len(v));w[seeds]=values
    w[free]=spsolve(lap[free][:,free],-lap[free][:,seeds]@values)
    if w.min()<-1e-8 or w.max()>1+1e-8:raise ValueError('Out-of-range harmonic binding')
    return np.clip(w,0.,1.)
def run(w,cap,package,audit,exe,folder):
    meshes,names,identity=load_registered(w,cap,package,audit);cfg=json.loads(package.read_text())['argv']
    payload=Path(cfg[cfg.index('--soft-tissue-payload')+1])
    poses=np.array(next(json.loads(row.partition('=')[2]) for row in (cap/'emitted-native/stderr.log').open() if row.startswith('registered_ligament_actual_pose_json='))['bodyPoses'])
    data=decode(payload,[29,30],poses);report=json.loads((folder/'binding-parity.json').read_text())
    variants={};rest={};stats={}
    for sid in (29,30):
        source,term,ww,b,faces=data[sid]
        v=meshes[51005,sid][0].astype(float);w0=ww[:,1];orig=term[:,0];dst=term[:,1]
        if np.max(np.linalg.norm(orig+(dst-orig)*w0[:,None]-v,axis=1))>2.5e-7:raise ValueError('Native replay mismatch')
        sites=report[str(sid)]['actuator_sites_world_m']
        distances=[np.linalg.norm(term[:,slot,None,:]-np.array([x for k,x in sites.items() if k.endswith(tag)])[None,:,:],axis=2) for slot,tag in enumerate(('origin','insertion'))]
        # One connected one-ring around each registered route endpoint.
        ring=[]
        for d in distances:
            vertex=d.argmin(0)
            ring.append(np.unique(faces[np.any(np.isin(faces,vertex),axis=1)]))
        pure0=np.flatnonzero(w0<1e-7);pure1=np.flatnonzero(w0>1-1e-7)
        anchor0=np.union1d(pure0,ring[0]);anchor1=np.union1d(pure1,ring[1])
        conflicts=np.intersect1d(anchor0,anchor1)
        if len(conflicts):raise ValueError(('Conflicting attachment owners',sid,conflicts))
        seeds=np.r_[anchor0,anchor1];values=np.r_[np.zeros(len(anchor0)),np.ones(len(anchor1))]
        harm=harmonic(source,faces,seeds,values)
        d0=distances[0].min(1);d1=distances[1].min(1);ratio=d0/(d0+d1)
        weights={'source':w0,'pelvis_only_diagnostic':np.zeros(len(v)),'attachment_harmonic':harm,
                 'attachment_distance':ratio**2/(ratio**2+(1-ratio)**2)}
        weights['attachment_distance'][seeds]=values
        for label,weight in weights.items():
            shape=v if label=='source' else (orig+(dst-orig)*weight[:,None]).astype('f4')
            variants.setdefault(label,{})[sid]=(shape,faces)
        rest[sid]=(source,faces)
        stats[sid]=dict(origin_minimum_distances_m=distances[0].min(0).tolist(),insertion_minimum_distances_m=distances[1].min(0).tolist(),pure_vertices=[len(pure0),len(pure1)],boundary_vertices=[len(anchor0),len(anchor1)])
        np.savez_compressed(folder/str(sid)/'weights.npz',**weights)
    result=dict(source=identity,registration=stats,source_reference_crossing_pairs=pair_count(exe,*rest[29],*rest[30]),variants={})
    for label,geometry in variants.items():
        trial_meshes=dict(meshes)
        for sid,vf in geometry.items():trial_meshes[51005,sid]=vf
        bilateral=pair_count(exe,*geometry[29],*geometry[30])
        row=dict(bilateral_crossings=bilateral)
        if label!='pelvis_only_diagnostic':
            row['sides']={}
            for sid in (29,30):
                orig,f=meshes[51005,sid];new=geometry[sid][0]
                pairs=interface_audit((51005,sid),trial_meshes,exe)
                base=json.loads((folder/str(sid)/'intersections.json').read_text())
                def keys(d):return {(k,i,j) for k,rr in d.items() for i,j in rr}
                old,now=keys(base),keys(pairs)
                row['sides'][sid]=dict(counts={k:len(rr) for k,rr in pairs.items()},new_pairs=len(now-old),cleared_pairs=len(old-now),max_change_mm=float(np.linalg.norm(orig-new,axis=1).max()*1000),volume_fraction=mesh_volume(new,f)/mesh_volume(orig,f))
        result['variants'][label]=row
    (folder/'fit.json').write_text(json.dumps(result,indent=2)+'\n')
    print([(k,v['bilateral_crossings']) for k,v in result['variants'].items()])
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
