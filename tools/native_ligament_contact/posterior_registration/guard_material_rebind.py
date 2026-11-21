"""Check the exact scope and identities of a material-registration update."""
import hashlib
import numpy as np
def digest(b):return hashlib.sha256(b).hexdigest()
def validate(old,new,nr,nb,nv,rows,lineage):
    bind=64+32*nr;base=bind+36*nb
    if len(old)!=len(new) or old[:bind]!=new[:bind] or old[base+nv*56:]!=new[base+nv*56:]:raise ValueError('Changed payload structure or topology')
    fields=lineage.get('neutral_material_bind',{})
    if fields and set(fields)!={'29','30'}:raise ValueError('Unexpected material rebind')
    bindings=np.frombuffer(old,'<u4',nb*9,bind).reshape(-1,9);after=np.frombuffer(new,'<u4',nb*9,bind).reshape(-1,9)
    u=np.frombuffer(old,'<u4',14*nv,base).reshape(-1,14);v=np.frombuffer(new,'<u4',14*nv,base).reshape(-1,14)
    affected=set();used=np.zeros(nb,int);vertices=np.zeros(nv,bool)
    for first,count,beg,n,_,_,sid,_ in rows.astype(int):
        used[first:first+count]+=1
        if sid not in (29,30):continue
        vertices[beg:beg+n]=True
        if not fields:continue
        field=fields[str(sid)];k=first+1;affected.add(k);subset=v[beg:beg+n];f=subset.view('<f4')
        if count!=2 or field['donor_ids']!=[128,131 if sid==29 else 145] or after[first:first+2,0].tolist()!=field['donor_ids']:raise ValueError('Unexpected material owners')
        if digest(after[k,1:].tobytes())!=field['binding_sha256'] or digest(subset.tobytes())!=field['vertex_field_sha256']:raise ValueError('Unqualified geometry or bindings')
        if not np.isfinite(f[:,:6]).all() or not np.isfinite(f[:,10:]).all() or np.any(f[:,10:]<0) or np.max(np.abs(f[:,10:].sum(1)-1))>2e-7:raise ValueError('Invalid blend field')
    if np.any(u[~vertices]!=v[~vertices]):raise ValueError('Changed unrelated tissue vertices')
    if not fields and np.any(u[:,6:]!=v[:,6:]):raise ValueError('Changed fixed material weights')
    if affected and np.any(used[list(affected)]!=1):raise ValueError('Material donor shared across records')
    allowed=np.zeros(nb,bool);allowed[list(affected)]=True
    if np.any(bindings[~allowed]!=after[~allowed]) or np.any(bindings[:,0]!=after[:,0]):raise ValueError('Changed unrelated body transforms')
