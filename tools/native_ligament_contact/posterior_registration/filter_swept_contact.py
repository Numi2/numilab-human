"""Conservatively cull triangle pairs separated on a fixed axis throughout each sampled linear sweep."""
from pathlib import Path
import json,sys
import numpy as np
from audit_finite_cortical_crossings import load_vertices
from material_context import sha

def narrow(assets,pairs,captures,out):
    d=json.loads((assets/'manifest.json').read_text());sides=d['closed_reverse']['sides']
    cf=np.fromfile(assets/'cortical-certificate-faces.bin','u4').reshape(-1,4)[:,:3];mf=np.fromfile(assets/'closed-material-triangles.bin','u4').reshape(-1,3,4)[:,0,:3]
    counts={'material_vertex':sum(r['vertices'] for r in sides),'cortical_vertex':d['closed_reverse']['unique_vertices']}
    states=[load_vertices(c,counts) for c in captures]
    states.sort(key=lambda r:r[0])
    n0,old=states[-2];n1,new=states[-1]
    states.append((n1+2,{k:x+2*(x-old[k])/(n1-n0) for k,x in new.items()}))
    rows=np.load(pairs);selected=[]
    for start in range(0,len(rows),4096):
        r=rows[start:start+4096];b=cf[r[:,0]];m=mf[r[:,1]]
        need=np.zeros(len(r),bool)
        for (_,s),(_,t) in zip(states,states[1:]):
            A0=s['material_vertex'][m];B0=s['cortical_vertex'][b];A1=t['material_vertex'][m];B1=t['cortical_vertex'][b]
            # Common translation cancels in relative interval separation.
            midA=.5*(A0+A1);midB=.5*(B0+B1)
            e=midA[:,[1,2,0]]-midA;f=midB[:,[1,2,0]]-midB
            a=np.cross(e[:,0],e[:,1]);c=np.cross(f[:,0],f[:,1])
            axes=[a,c,*[np.cross(e[:,i],f[:,j]) for i in range(3) for j in range(3)],
                *[np.cross(a,e[:,i]) for i in range(3)],*[np.cross(c,f[:,i]) for i in range(3)]]
            separated=np.zeros(len(r),bool)
            for axis in axes:
                norm=np.linalg.norm(axis,axis=1);valid=norm>1e-28
                for left,right,L,R in ((A0,B0,A1,B1),(B0,A0,B1,A1)):
                    def gap(l,h):
                        origin=l[:,:1];return np.einsum('ijk,ik->ij',h-origin,axis).min(1)-np.einsum('ijk,ik->ij',l-origin,axis).max(1)
                    # A scale-relative 50 nm authoring search band covers published coordinate
                    # rounding by a margin; it is never a force offset or admission tolerance.
                    separated|=valid&(gap(left,right)>5.e-8*norm)&(gap(L,R)>5.e-8*norm)
            need|=~separated
        selected.append(r[need])
    selected=np.vstack(selected);np.save(out,selected)
    out.with_suffix('.json').write_text(json.dumps(dict(parent_sha256=sha(pairs),source_sha256=[sha(x) for x in captures],
        observed_steps=[n for n,_ in states[:-1]],extrapolated_step=states[-1][0],search_band_m=5e-8,
        input_pairs=len(rows),retained=len(selected)),indent=2)+'\n')
if __name__=='__main__':
    a,p,out=map(Path,sys.argv[1:4]);narrow(a,p,list(map(Path,sys.argv[4:])),out)
