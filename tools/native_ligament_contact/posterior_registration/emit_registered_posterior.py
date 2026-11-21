"""Apply certified source-coordinate edits to a separate NHTISS4 candidate with the same bindings."""
import json,struct,sys
from pathlib import Path
import numpy as np
from material_context import sha

def emit(cfg_path,patches,out,kind):
    cfg=json.loads(cfg_path.read_text());args=cfg['argv'];at=args.index('--soft-tissue-payload')+1
    old=Path(args[at]);m=json.loads(old.with_suffix('.manifest.json').read_text())
    assert sha(old)==m['payload']['sha256']
    data=bytearray(old.read_bytes());magic,abi,nr,nb,nv,ni,_,_=struct.unpack_from('<8s6I32s',data)
    if (magic,abi)!=(b'NHTISS4\0',5):raise ValueError('Wrong muscle payload')
    rr=np.ndarray((nr,8),'<u4',data,64);base=64+32*nr+36*nb
    vf=np.ndarray((nv,14),'<f4',data,base)
    ii=np.ndarray((ni,),'<u4',data,base+56*nv)
    result={}
    for sid in (29,30):
        _,_,start,n,begin,nidx,_,_=next(r for r in rr.astype(int) if r[6]==sid)
        v=vf[start:start+n,:3];normals=vf[start:start+n,3:6];f=ii[begin:begin+nidx].reshape(-1,3).astype(int)
        if f.min()>=start:f-=start
        assert f.max()<n
        p=patches/str(sid)/(kind+'.npz');edit=np.load(p);ids=edit['indices'];pins=edit['fixed_site_vertices']
        before=v.copy();prior_weights=vf[start:start+n,6:].tobytes()
        assert not np.intersect1d(ids,pins).size
        v[ids,0]=edit['source_x']
        near=np.any(np.isin(f,ids),axis=1)
        refresh=np.unique(f[near])
        def vector_areas(x):
            a,b,c=x.astype(float)[f].transpose(1,0,2)
            return np.cross(b-a,c-a)
        A=vector_areas(before);B=vector_areas(v)
        a=np.zeros((n,3));b=np.zeros((n,3))
        for j in range(3):np.add.at(a,f[:,j],A);np.add.at(b,f[:,j],B)
        # Rotate the authored vertex normal by the change in area-weighted normal.
        # Adjacent material vertices preserve their original normal convention.
        oldnorm=a[refresh];newnorm=b[refresh]
        oldnorm/=np.linalg.norm(oldnorm,axis=1)[:,None];newnorm/=np.linalg.norm(newnorm,axis=1)[:,None]
        n0=normals[refresh].astype(float);axis=np.cross(oldnorm,newnorm);dot=(oldnorm*newnorm).sum(1)
        if not (dot>-.999).all():raise ValueError('Ambiguous material normal flip')
        n1=n0+np.cross(axis,n0)+np.cross(axis,np.cross(axis,n0))/(1+dot[:,None])
        n1/=np.linalg.norm(n1,axis=1)[:,None];normals[refresh]=n1
        assert prior_weights==vf[start:start+n,6:].tobytes()
        result[sid]=dict(edit=sha(p),source_vertices_changed=len(ids),fixed_attachment_neighborhood_vertices=len(pins),normals_updated=len(refresh))
    out.mkdir(exist_ok=True);candidate=out/old.name;candidate.write_bytes(data)
    m['source']['gluteal_registration']={'parent_payload_sha256':sha(old),'source_coordinate_edits':result,'scope':'Registration trial at exact source connectivity. Existing MyoSim and blood owners unchanged. Gluteal continuum not installed.'}
    m['payload']['sha256']=sha(candidate)
    manifest=candidate.with_suffix('.manifest.json');manifest.write_text(json.dumps(m,separators=(',',':'))+'\n')
    args[at]=str(candidate.resolve())
    cfg['hashes'][str(candidate.resolve())]=sha(candidate);cfg['hashes'][str(manifest.resolve())]=sha(manifest)
    cfg['hashes'][str(cfg_path)]=sha(cfg_path)
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (out/'source.json').write_text(json.dumps(dict(parent=str(old),parent_sha256=sha(old),candidate_sha256=sha(candidate),edits=result),indent=2)+'\n')
if __name__=='__main__':emit(*map(Path,sys.argv[1:4]),sys.argv[4])
