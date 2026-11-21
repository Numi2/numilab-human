"""Test the actual emitted bone banks against every neighbor on recorded poses."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation as R
from shoulder_inputs import old,load,digest
from source_pair_audit import inputs
from bone_overlap import intersections
from audit_solids import winding
from fast_skin_intersection_audit import install_for_float32

def components(v,f):
    uniq,inv=np.unique(v,axis=0,return_inverse=True); tris=inv[f]
    e=tris[:,[0,1,1,2,2,0]].reshape(-1,2)
    g=coo_matrix((np.ones(len(e),np.int8),e.T),shape=(len(uniq),len(uniq)))
    _,labels=connected_components(g,directed=False)
    seen=np.unique(labels[tris[:,0]])
    return [np.unique(f[labels[tris[:,0]]==i]) for i in seen]

def verify(launch,ref,motion,bones,out):
    _,poses,default,_=inputs(launch,ref,motion)
    cfg,cat,ms,_,_=load(launch,ref,motion)
    old.audit.ag.BONES=bones
    _,raw,_,_,_,_,_,newcat,_,vo,io=old.audit.ag.read_payloads()
    d=json.loads(bones.with_suffix('.manifest.json').read_text());assert digest(bones)==d['payload']['sha256']
    changed={row['stable_id'] for row in d['source']['connected_girdle_parent']['changed_anchors']}
    new=old.load_local_meshes(raw,newcat,vo,io)
    assert len(new)==185 and changed=={8,9,10,11}
    for sid in ms:
        assert np.array_equal(ms[sid][1],new[sid][1])
        if sid not in changed:assert np.array_equal(ms[sid][0],new[sid][0])
    comp={s:components(v,f) for s,(v,f,*_) in new.items()}
    install_for_float32();rows=[];pair_visits=0;containment_probes=0
    # Transform bounding-box corners first, then materialize only banks whose
    # conservative world boxes overlap a changed bank. Include rounding padding.
    corners={s:np.array(np.meshgrid(*zip(v.min(0),v.max(0)))).reshape(3,-1).T for s,(v,*_) in new.items()}
    for step,p in [(0,default),*sorted(poses.items())]:
        owners={b:(R.from_quat(p[b][3:]),p[b][:3]) for b in {r['body'] for r in cat.values()}}
        def world(s,x):
            rr,t=owners[cat[s]['body']];return rr.apply(x)+t
        box={s:np.stack([a.min(0)-2e-6,a.max(0)+2e-6]) for s,x in corners.items() for a in [world(s,x)]}
        pts={};fail=[]
        for s in sorted(changed):
            for k in cat:
                if k==s or k in changed and k<s:continue
                if np.any(box[s][1]<box[k][0]) or np.any(box[k][1]<box[s][0]):continue
                for n in (s,k):
                    if n not in pts:pts[n]=world(n,new[n][0]).astype(np.float32)
                pair_visits+=1;a,b=pts[s],pts[k]
                aa,ab=a.min(0),a.max(0);ba,bb=b.min(0),b.max(0)
                if np.any(ab<ba) or np.any(bb<aa):continue
                pairs=intersections(a,new[s][1],b,new[k][1])
                if pairs:fail.append(dict(banks=[s,k],pairs=len(pairs),first=pairs[0]));continue
                for u,v,other,lo,hi in [(s,k,a,ba,bb),(k,s,b,aa,ab)]:
                    for ids in comp[u]:
                        x=other[ids]
                        if np.all(x.min(0)>=lo) and np.all(x.max(0)<=hi):
                            containment_probes+=1
                            w=winding(pts[v][new[v][1]].astype(float),x[0].astype(float))
                            if abs(w)>.5:fail.append(dict(contained=[u,v],vertex=int(ids[0]),winding=w))
        rows.append(dict(step=step,failures=fail))
        if step%800==0 or fail:print(step,fail,flush=True)
    result=dict(bone_sha256=digest(bones),pose_csv_sha256=digest(motion),changed_banks=sorted(changed),bank_count=len(cat),
        pose_count=len(rows),broad_phase_visits=pair_visits,contained_component_probes=containment_probes,
        total_crossing_pairs=sum(f.get('pairs',0) for rr in rows for f in rr['failures']),
        contained_components=sum('contained' in f for rr in rows for f in rr['failures']),poses=rows)
    out.write_text(json.dumps(result,indent=2)+'\n')
    if any(rr['failures'] for rr in rows):raise SystemExit(1)
if __name__=='__main__':verify(*map(Path,sys.argv[1:]))
