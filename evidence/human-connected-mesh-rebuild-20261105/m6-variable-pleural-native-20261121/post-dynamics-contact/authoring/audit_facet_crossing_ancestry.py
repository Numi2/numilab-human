"""Classify exact GPU crossings by the original source faces on the two solids."""
import json,sys
from collections import Counter
from pathlib import Path
import numpy as np
from compare_hand_interfaces import query
from count_accepted_cortical_crossings import dyadic,plane,straddles
from audit_finite_cortical_crossings import load_vertices
from material_context import sha

def run(part,assets,capture,predicate,out,atlas=None):
    meta=json.loads((assets/'manifest.json').read_text());g=meta['closed_reverse']
    step,vert=load_vertices(capture,{'material_vertex':sum(s['vertices'] for s in g['sides']),'cortical_vertex':g['unique_vertices']})
    mf=np.fromfile(assets/'closed-material-triangles.bin','u4').reshape(-1,3,4)[:,0,:3].astype(int)
    T=np.fromfile(assets/'cortical-triangles.bin','f4').reshape(-1,10,4)[:,:3,:3]
    pool=np.fromfile(assets/'pool-material-points.bin','f4').reshape(-1,7,4)[g['first']:g['first']+g['unique_vertices']]
    dom=pool[:,3].copy().view('u4');bank=pool[:,4].copy().view('u4')
    candidates=None
    if atlas:
        with np.load(atlas) as data:candidates=set(tuple(map(int,row)) for row in data['pairs'][:,:4])
    coverage=Counter();missing=[]
    tally=Counter();per={};hist={};witness={}
    for side,(s,fem) in enumerate(zip(g['sides'],(131,145))):
        sid=29+side;auth=np.load(part/'conforming-boundary'/f'{sid}.npz');port=np.load(part/'ports'/f'{sid}.npz')
        i=s['first_vertex'];n=s['vertices'];f0=s['first_face'];v=vert['material_vertex'][i:i+n];F=mf[f0:f0+s['faces']]-i
        mapping={tuple(f):k for k,f in enumerate(auth['face'])};src=auth['source_face'][[mapping[tuple(f)] for f in F]]
        for body in (1,128,fem):
            ix=np.flatnonzero((dom[:,1]==fem)&(dom[:,2]==body));a,m=np.unique(bank[ix,:2],axis=0)[0];local={tuple(p):j for j,p in enumerate(pool[ix,2,:3])}
            C=np.array([[local[tuple(p)] for p in tri] for tri in T[a:a+m]],int);b=vert['cortical_vertex'][ix];vertices=np.vstack((v,b));faces=np.vstack((F,C+n));xyz=dyadic(vertices)
            pairs=query(predicate,vertices,faces,np.arange(len(faces))<len(F))
            for j,k in sorted(pairs):
                if not(j<len(F)<=k):continue
                bf=k-len(F);A=[xyz[z] for z in F[j]];B=[xyz[n+z] for z in C[bf]]
                if not(straddles(B,A[0],plane(*A)) and straddles(A,B[0],plane(*B))):continue
                idx=src[j];parent=-1
                if idx>=len(port['target_bank']):reason='original muscle exterior'
                elif int(port['target_body'][idx])!=body:reason='different anatomical interface'
                else:
                    parent=int(port['target_face'][idx]);shared=len(set(C[parent])&set(C[bf]));reason={3:'registered parent',2:'shared cortical edge',1:'shared cortical vertex',0:'nonincident cortical face'}[shared]
                if candidates is not None:
                    key=(side,mapping[tuple(F[j])],bf,body);present=key in candidates
                    coverage['covered' if present else 'missing']+=1
                    if not present and len(missing)<30:missing.append([*key,reason])
                mface=v[F[j]];bface=b[C[bf]];nb=np.cross(bface[1]-bface[0],bface[2]-bface[0]);nb/=np.linalg.norm(nb)
                z=(mface-bface[0])@nb;arc=min(-z.min(),z.max())
                tally[reason]+=1;per.setdefault(f'{sid}/{body}',Counter())[reason]+=1;hist.setdefault(reason,[]).append(arc)
                if reason not in witness or witness[reason]['smaller_plane_span_m']<arc:
                    witness[reason]=dict(muscle=sid,cortical_body=body,material_face=int(j+f0),queried_cortical_face=int(bf),registered_cortical_parent=parent,smaller_plane_span_m=arc)
    out.write_text(json.dumps(dict(accepted_step=step,source=sha(capture),atlas_sha256=sha(atlas) if atlas else None,atlas_coverage=coverage,missing_examples=missing,counts=tally,regional=per,largest_witness=witness,median_smaller_plane_span_m={k:float(np.median(v)) for k,v in hist.items()}),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
