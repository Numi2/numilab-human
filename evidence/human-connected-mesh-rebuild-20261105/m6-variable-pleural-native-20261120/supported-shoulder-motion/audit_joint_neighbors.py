"""Check all changed arm-branch versus other-arm or trunk bone interfaces."""
from pathlib import Path
import json,struct,sys
import numpy as np
from scipy.spatial.transform import Rotation
from shoulder_inputs import load
from bone_overlap import intersections
from fast_skin_intersection_audit import install_for_float32


def branches(raw,links):
    nb,nj=struct.unpack_from('<2I',raw,20)
    parent={i:[] for i in range(nb)}
    for j in range(nj):
        p,c=struct.unpack_from('<2I',raw,224+160*nb+144*j)
        parent[p].append(c)
    def descendants(r):
        return {r}.union(*(descendants(ch) for ch in parent[r]))
    sets=[set.union(*(descendants(k) for k in parent[p])) for p in links]
    assert not sets[0]&sets[1] and all(sets)
    return sets


def audit(launch,anchors,motion,frame):
    config,catalog,meshes,spec,poses=load(launch,anchors,motion);cal=json.loads(frame.read_text())
    parents=[catalog[a['stable_ids'][0]]['body'] for a in spec['candidates']]
    sets=branches(Path(config['argv'][1]).read_bytes(),parents)
    assert all(catalog[a['stable_ids'][1]]['body'] in ids for a,ids in zip(spec['candidates'],sets))
    install_for_float32(); report=[]
    for step in (-3,32,256,8000):
        tagged={}
        for sid,(v,f,*_) in meshes.items():
            b=catalog[sid]['body']; pose=poses[step][b]
            g=next((i for i,bs in enumerate(sets) if b in bs),-1)
            world=pose[:3]+Rotation.from_quat(pose[3:]).apply(v)
            new=world
            if g>=0:
                p=poses[step][parents[g]];rs=Rotation.from_quat(p[3:])
                delta=Rotation.from_quat(cal[g]['parent_rotation_xyzw']);dx=np.array(cal[g]['parent_translation_base_m'])
                assert cal[g]['trials'][-1]['additional_shift_mm']==0
                new=p[:3]+rs.apply(delta.apply(rs.inv().apply(world-p[:3]))+dx)
            tagged[sid]=(g,f,[world,new],[(v.min(axis=0),v.max(axis=0)) for v in (world,new)])
        results=[]
        keys=sorted(tagged)
        for i,a in enumerate(keys):
            ga,fa,va,ba=tagged[a]
            for b in keys[:i]:
                gb,fb,vb,bb=tagged[b]
                if ga==gb:continue
                counts=[]; witness=[]
                for k in (0,1):
                    if np.all(ba[k][1]+1e-7>=bb[k][0]) and np.all(bb[k][1]+1e-7>=ba[k][0]):
                        hits=intersections(va[k],fa,vb[k],fb)
                    else:hits=[]
                    counts.append(len(hits));witness.append(hits[:1])
                if any(counts):results.append(dict(bones=[a,b],reference=counts[0],candidate=counts[1],first_pairs=witness))
        report.append(dict(step=step,rows=results))
        print(step,sum(x['reference'] for x in results),sum(x['candidate'] for x in results),
              [(x['bones'],x['candidate']) for x in results if x['candidate'] and not x['reference']],flush=True)
    return report
if __name__=='__main__':
    *args,out=map(Path,sys.argv[1:]);out.write_text(json.dumps(audit(*args),indent=2)+'\n')
