"""Carry the registered bilateral muscle shape into a common neutral pelvis/femur bind."""
import hashlib,json,struct,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pose_glutes import parts,model
from material_context import sha
from bind_posterior_receipt import run as bind

def unit(v):
    n=np.linalg.norm(v,axis=1,keepdims=True)
    if not np.isfinite(n).all() or n.min()<1e-10:raise ValueError('Missing material normal')
    return v/n
def emit(cfgpath,previous,fit,receipt,out):
    cfg=json.loads(cfgpath.read_text());args=cfg['argv'];at=args.index('--soft-tissue-payload')+1
    old=Path(args[at]);history=json.loads((previous/'source.json').read_text());source=previous/old.name
    if sha(old)!=history['parent_sha256'] or sha(source)!=history['candidate_sha256']:raise ValueError('Unverified registration source')
    evidence=json.loads((fit/'result.json').read_text())
    if evidence['parent']!=sha(source) or any(any(row[1:]) for row in evidence['all_pose_intersections']):raise ValueError('Unqualified motion fit')
    raw=bytearray(source.read_bytes());magic,version,nr,nb,nv,ni,_,_=struct.unpack_from('<8s6I32s',raw);assert (magic,version)==(b'NHTISS4\0',5)
    rr=np.ndarray((nr,8),'<u4',raw,64);start=64+nr*32;atv=start+36*nb
    v=np.ndarray((nv,14),'<f4',raw,atv);slots=v.view('<u4')
    pose={r['body_index']:np.array(r['position_m']+r['quaternion_xyzw']) for r in json.loads(receipt.read_text())['initial_anatomical_registration']['body_poses']}
    rec={}
    for sid in (29,30):
        fitrow=evidence['by_side'][str(sid)]
        if fitrow['inverted_cell_samples'] or fitrow['low_quality_cell_samples']:raise ValueError('Unqualified source cells')
        data=parts(source,sid);src,w,donors,_,aff=data
        fb,count,beg,n,_,_,_,_=next(r for r in rr.astype(int) if r[6]==sid)
        assert count==2 and donors==[128,131 if sid==29 else 145]
        arr=np.load(fit/(str(sid)+'.npz'));params=arr['secondary_binding'];ww=arr['weights']
        rots=[Rotation.from_quat(pose[b][3:]).as_matrix()@a[0]/np.cbrt(np.linalg.det(a[0])) for b,a in zip(donors,aff)]
        old_n=v[beg:beg+n,3:6].astype(float)
        world_n=unit(sum(w[:,j,None]*(old_n@r.T) for j,r in enumerate(rots)))
        field=v[beg:beg+n];field[:,:3]=arr['vertices'];field[:,3:6]=unit(world_n@rots[0]);slots[beg:beg+n,6:10]=[0,1,4294967295,4294967295];field[:,10:]=0;field[:,10:12]=ww
        if not np.array_equal(ww[arr['origins'],1],np.zeros(len(arr['origins']))) or not np.all(ww[arr['insertions'],1]==1.):raise ValueError('Anatomical sites changed donor')
        binding_offset=start+36*(fb+1)+4
        raw[binding_offset:binding_offset+32]=params.astype('<f4').tobytes()
        new_aff=[aff[0],(Rotation.from_quat(params[3:7]).as_matrix()*params[7],params[:3])]
        old_x=model(data,pose,1)[0]
        for b,(a,d) in zip(donors,new_aff):
            one=Rotation.from_quat(pose[b][3:]).apply(field[:,:3]@a.T+d)+pose[b][:3]
            if np.max(np.linalg.norm(one-old_x,axis=1))>1e-6:raise ValueError('Neutral registration moved')
        rec[sid]=dict(donor_ids=donors,fit_sha256=sha(fit/(str(sid)+'.npz')),binding_sha256=hashlib.sha256(params.astype('<f4').tobytes()).hexdigest(),vertex_field_sha256=hashlib.sha256(field.tobytes()).hexdigest(),minimum_reference_normal_cosine=float(np.min(np.sum(unit(field[:,3:6]@rots[0].T)*world_n,axis=1))),pelvic_site_vertices=arr['origins'].tolist(),femoral_site_vertices=arr['insertions'].tolist())
    out.mkdir(exist_ok=True);new=out/old.name;new.write_bytes(raw)
    lineage=dict(parent=str(old),parent_sha256=sha(old),geometry_parent_sha256=sha(source),candidate_sha256=sha(new),reference_receipt_sha256=sha(receipt),trajectory_qualification_sha256=sha(fit/'result.json'),neutral_material_bind=rec)
    (out/'source.json').write_text(json.dumps(lineage,indent=2)+'\n')
    m=json.loads(source.with_suffix('.manifest.json').read_text());m['source']['registered_gluteal_donor_reference']=lineage;m['payload']['sha256']=sha(new);mp=new.with_suffix('.manifest.json');mp.write_text(json.dumps(m,separators=(',',':'))+'\n')
    args[at]=str(new.resolve());cfg['hashes'].update({str(new.resolve()):sha(new),str(mp.resolve()):sha(mp),str(cfgpath):sha(cfgpath)})
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    bind(out)
if __name__=='__main__':emit(*map(Path,sys.argv[1:]))
