"""Measure exact exposed interfaces on both complete native capture packs."""
import json,mmap,runpy,struct
from pathlib import Path
import numpy as np
from prepare_posterior_muscles import interface_audit
def inspect_native_neighbors(pack,rows,audit_root,exe):
    frames={r['body_index']:r for r in rows}
    reader=runpy.run_path(str(audit_root/'audit.py'));reader['BANK_CLASSES'].add(51007)
    banks={};omitted=[];ambiguous=[]
    with pack.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as raw:
        c=reader['clearance'];hh=c._PACK_HEADER;dd=c._PACK_DIRECTORY
        magic,version,count,*_=hh.unpack_from(raw);assert (magic,version)==(b'MRVPACK2',2)
        sections={d[0]:d for d in [dd.unpack_from(raw,hh.size+j*dd.size) for j in range(count)]}
        vertices,indices,primitives,instances=[sections[k] for k in (2,3,4,5)]
        xyz=np.ndarray((vertices[4],20),'<f4',raw,vertices[2])[:,:3];tri=np.ndarray(indices[4],'<u4',raw,indices[2])
        for i in range(primitives[4]):
            first,count,_,inst,semantic,_,_,sid=struct.unpack_from('<8I',raw,primitives[2]+i*primitives[5]);key=(semantic,sid)
            if semantic not in reader['BANK_CLASSES']:continue
            tr=struct.unpack_from('<8f12I',raw,instances[2]+inst*instances[5])
            if tr[10]==3 and tr[9] not in frames:omitted.append([*key,tr[9]]);continue
            if key in banks:ambiguous.append(list(key));continue
            t=tri[first:first+count].reshape(-1,3);used=np.unique(t);x=reader['world_positions'](xyz[used],tr)
            if tr[10]==3:
                p=frames[tr[9]];x=reader['world_positions'](x,(*p['position_m'],1.,*p['quaternion_xyzw']))
            elif tr[10]!=0:raise ValueError('Unresolved native instance')
            banks[key]=(x.copy(),np.searchsorted(used,t))
    return dict(sides={sid:{key:len(val) for key,val in interface_audit((51005,sid),banks,exe).items()} for sid in (29,30)},available_surfaces=len(banks),missing_frame_surfaces=omitted,duplicate_bank_surfaces=ambiguous)
