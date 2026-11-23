"""Measure compiled shoulder pairs on retained body poses using the physical shoulder parent chain."""
from pathlib import Path
import json,sys
from itertools import combinations
import numpy as np
from scipy.spatial.transform import Rotation
from audit_registered_skeleton import HEADER,RECORD,closed_solid,intersection_mm3,digest
from compile_connected_joint import matrix
from shoulder_inputs import load


def run(job, anchors, motion, compiled, bones, output):
    from source_pair_audit import inputs
    launch=job/'launch.json';cfg,cat,ms,_,poses=load(launch,anchors,motion)
    _,_,rest,_=inputs(launch,anchors,motion)
    payload=Path(next(bones.glob('*.nhbones')));raw=payload.read_bytes()
    rb=(compiled/'source-bound.nhrigid').read_bytes()
    count,nvert=HEADER.unpack_from(raw)[2:4];vo=HEADER.size+count*RECORD.size;io=vo+nvert*24
    groups={};control={};candidate={};ids={};f={};frames={}
    for b,frame in zip((84,34),json.loads((compiled/'frames.json').read_text())):
        frames[b]=matrix(frame['parent_translation_base_m']+frame['parent_rotation_xyzw'])
    for i in range(count):
        rec=RECORD.unpack_from(raw,HEADER.size+RECORD.size*i)
        b,start,n,first,nt,sid=rec[:6]
        x=np.frombuffer(raw,'<f4',6*n,vo+24*start).reshape(-1,6)[:,:3]
        candidate[sid]=Rotation.from_quat(rec[9:13]).apply(rec[13]*x.astype(float))+rec[6:9]
        control[sid]=ms[sid][0];f[sid]=ms[sid][1]
        actual=np.frombuffer(raw,'<u4',nt,io+4*first).reshape(-1,3).astype('i8')-start
        if not np.array_equal(actual,f[sid]):raise ValueError('Unexpected cortical topology change')
        ids[sid]=b
        while b:
            if b-1 in frames:groups[sid]=b-1;break
            parent,child=__import__('struct').unpack_from('<2I',rb,224+157*160+(b-1)*144)
            if child!=b or parent>=b:raise ValueError('Disconnected joint')
            b=parent
    # Include the complete shoulder girdle and both upper arms against every
    # other region. Stable bone IDs here are pinned by the launch manifest.
    scope={8,9,10,11,12,13}
    pairs=[(a,b) for a,b in combinations(ids,2) if (a in scope or b in scope) and ids[a]!=ids[b]]
    def corners(v):
        return np.stack(np.meshgrid(*zip(v.min(0),v.max(0))),axis=-1).reshape(8,3)
    local_boxes=[{k:corners(v) for k,v in src.items()} for src in (control,candidate)]
    rows=[];violations={};closed_counts=[0,0];keys=((10,12),(11,13))
    for step,p in [(0,rest),*sorted(poses.items())]:
        W={b:matrix(p[b]) for b in set(ids.values())|set(frames)}
        move={b:W[b]@J@np.linalg.inv(W[b]) for b,J in frames.items()}
        transforms=[{s:W[b] for s,b in ids.items()},{s:(move[groups[s]]@W[b] if s in groups else W[b]) for s,b in ids.items()}]
        volumes=[]
        for k,(src,tr) in enumerate(zip((control,candidate),transforms)):
            mats=np.array([tr[s] for s in ids]); boxes=np.array([local_boxes[k][s] for s in ids])
            v=boxes@mats[:,:3,:3].transpose(0,2,1)+mats[:,None,:3,3]
            lo=v.min(1)-2e-7;hi=v.max(1)+2e-7
            keylist=np.array(pairs)-1;a,b=keylist.T
            cand=keylist[np.all((hi[a]>=lo[b])&(hi[b]>=lo[a]),axis=1)]+1
            solids={};vv={}
            for a,b in cand:
                for s in (a,b):
                    if s not in solids:
                        A=tr[s];x=(src[s]@A[:3,:3].T+A[:3,3]).astype('f4').astype(float)*1000
                        solids[s]=closed_solid(x,f[s])
                overlap=intersection_mm3(solids[a],solids[b])
                if overlap>1e-9:vv[int(a),int(b)]=overlap
            closed_counts[k]+=len(cand);volumes.append(vv)
        old,new=volumes
        for pair in set(new)-set(old):
            witness=violations.setdefault(str(pair),dict(first_step=int(step),poses=0,max_mm3=0.))
            witness['poses']+=1;witness['max_mm3']=max(witness['max_mm3'],new[pair])
        rows.append(dict(step=int(step),original_pairs=len(old),candidate_pairs=len(new),
                         shoulders_before=[old.get(k,0.) for k in keys],shoulders_after=[new.get(k,0.) for k in keys]))
    report=dict(launch=digest(launch),rigid=digest(compiled/'source-bound.nhrigid'),bones=digest(payload),
                motion=digest(motion),scope=sorted(scope),pairs_per_pose=len(pairs),total_poses=len(rows),narrow_visits=closed_counts,new_pairs=violations,poses=rows)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(len(rows),'new pairs',violations,'shoulder maxima',[max(x['shoulders_after'][i] for x in rows) for i in range(2)])


if __name__=='__main__':run(*map(Path,sys.argv[1:]))
