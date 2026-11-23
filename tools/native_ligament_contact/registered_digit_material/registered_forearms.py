"""Pose BRD and BRD_l through their actual NHTISS4 surface owners and named MyoSim sites."""
import json
import struct
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation
from extensor_registration import apply, affines, landmarks, sha, tissue
from solve_extensor_sites import bind_dofs

def read(path):
    return json.loads(path.read_text())

def records(data):
    magic,abi,ns,nb,nv,ni,_=struct.unpack_from('<8s6I',data)
    if (magic,abi)!=(b'NHTISS4\0',5):raise ValueError('NHTISS4 version differs')
    return np.frombuffer(data,'<u4',8*ns,64).reshape(ns,8),64+32*ns+36*nb,nv

def native_sites(raw,name,index,expected,tendons):
    magic,abi,nb,nm,ns,nw,nr,nt,na,size,_=struct.unpack_from('<8s9I32s',raw)
    ri=76+16*ns+64*nw;mi=ri+16*nr
    if magic!=b'NHMYO2\0\0' or abi!=2 or len(raw)!=mi+nm*(164+32) or index>=nm:raise ValueError('Wrong MyoSim ABI or index')
    _,start,count,res=struct.unpack_from('<4I',raw,mi+164*index)
    if count!=expected or res or start+count>nr:raise ValueError('Changed force route')
    sites=[]
    for off in range(count):
        kind,idx,_,_=struct.unpack_from('<4I',raw,ri+16*(start+off))
        if kind!=1:continue
        if idx>=ns:raise ValueError('Route site index')
        body,*local=struct.unpack_from('<I3f',raw,76+16*idx)
        if off in (0,count-1):
            ep=tendons[(name,'origin' if off==0 else 'insertion')]
            if idx!=ep['source_site_index'] or body!=ep['body_index'] or np.linalg.norm(np.subtract(local,ep['source_local_point_m']))>1e-10:raise ValueError('Different tendon registration')
            local=ep['resolved_local_point_m']
        sites.append(dict(body=body,local_m=local,site_index=idx))
    if len(sites)<2:raise ValueError('Missing physical origin or insertion')
    return sites

class Muscle:
    def __init__(self,job,sid):
        args=job['argv']
        def pin(p):
            p=Path(p)
            if job['pins'][str(p)]!=sha(p):raise ValueError('Different pinned input '+p.name)
            return p
        self.child=pin(args[args.index('--soft-tissue-payload')+1])
        self.receipt=pin(args[args.index('--resting-anatomy-receipt')+1])
        data=self.child.read_bytes()
        self.sid=sid
        meta=read(self.child.with_suffix('.manifest.json'))
        if meta['payload']['sha256']!=sha(self.child):raise ValueError('Child manifest mismatch')
        entry=next(s for s in meta['source']['surfaces'] if s['stable_id']==sid and s['layer']=='muscle')
        route,=entry['matched_muscles'];self.name=entry['label']
        myo=pin(args[2]);tendon=Path(args[args.index('--tendon-payload')+1]);maps=read(tendon.with_suffix('.manifest.json'))
        if maps['payload']['sha256']!=sha(tendon) or maps['source']['myosim_muscle_payload_sha256']!=sha(myo):raise ValueError('Wrong physical route owner')
        tags={(k['muscle'],k['endpoint']):k for k in maps['endpoints']}
        self.sites=native_sites(myo.read_bytes(),route['name'],route['source_actuator_index'],route['source_route_node_count'],tags)
        parent=read(self.receipt)['passive_tissue_parent_bindings']
        if parent['child_payload_sha256']!=sha(self.child):raise ValueError('Stale material ancestry')
        matches=[r for r in parent['surfaces'] if r['stable_id']==sid]
        self.owner=self.child;self.bank=sid;self.map=None;self.parent_index=None
        tab,vo,nv=records(data);row,=tab[tab[:,6]==sid]
        idx=np.frombuffer(data,'<u4',int(row[5]),vo+56*nv+4*int(row[4])).reshape(-1,3)
        ids,self.faces=np.unique(idx,return_inverse=True);self.faces=self.faces.reshape(-1,3)
        if ids.min()<row[2] or ids.max()>=row[2]+row[3]:raise ValueError('Child index out of range')
        if matches:
            item,=matches;verts=item['vertices']
            keys={(r[0],r[1]) for v in verts for r in v}
            if len(keys)!=1 or len(verts)!=row[3]:raise ValueError('Requires a multiple-owner registration')
            self.parent_index,self.bank=keys.pop();ref=parent['source_payloads'][self.parent_index]
            self.owner=Path(ref['path']);data=self.owner.read_bytes()
            if sha(self.owner)!=ref['sha256']:raise ValueError('Wrong source parent')
            rows=[verts[int(i)-int(row[2])] for i in ids];ptr=np.r_[0,np.cumsum([len(r) for r in rows])]
            ds=[t for v in rows for t in v];tt,_,_=records(data);rr,=tt[tt[:,6]==self.bank]
            self.map=csr_matrix(([t[3] for t in ds],[t[2] for t in ds],ptr),shape=(len(rows),int(rr[3])))
            if self.map.data.min()<0 or np.max(abs(self.map.sum(1)-1))>2e-6:raise ValueError('Invalid material parent weights')
        tab,vo,nv=records(data);r,=tab[tab[:,6]==self.bank];bi,nb,vi,n,_,_,_,_=map(int,r)
        self.binding_offset=64+32*len(tab)+36*bi
        self.bindings=[struct.unpack_from('<I8f',data,self.binding_offset+36*i) for i in range(nb)]
        v=np.frombuffer(data,'<f4',14*nv,vo).reshape(nv,14)[vi:vi+n]
        if self.map is None:v=v[ids-vi]
        self.x=v[:,:3].astype(float);self.slots=v[:,6:10].copy().view('<u4');self.w=v[:,10:14].astype(float)
        self.slots[(self.slots==0xffffffff)&(self.w==0)]=0
        if self.slots.max()>=nb or self.w.min()<0 or abs(self.w.sum(1)-1).max()>1e-5:raise ValueError('Invalid donor weights')
    def positions(self,pose,rows):
        a,t=affines(self.slots,self.w,rows,pose);p=apply(a,self.x)+t
        if self.map is not None:p=self.map@p
        return p.astype('f4')
    def freedoms(self,pose):
        src,_,owner=landmarks(self.bindings,pose,[self.sites])
        return bind_dofs(self.bindings,src,owner)
    def site_error(self,rows):
        first={r[0]:r for r in self.bindings};second={r[0]:r for r in rows}
        def rot(q):return Rotation.from_quat(q[4:8])
        errors=[]
        for s in self.sites:
            old=first[s['body']];new=second[s['body']]
            loc=np.asarray(s['local_m']);x=rot(old).inv().apply((loc-old[1:4])/old[8])
            errors.append(np.linalg.norm(rot(new).apply(x)*new[8]+new[1:4]-loc))
        return float(max(errors))
