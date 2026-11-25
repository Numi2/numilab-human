"""Measure exterior distance and affected area on matched anatomical triangle samples."""
import json,mmap,sys
from pathlib import Path
import igl,numpy as np
from bindings import audit,read_skin

def run(cases,out):
    banks=set()
    for report,case in cases:
        r=json.loads(report.read_bytes())
        banks.update((b['semantic'],b['stable_id'],b['primitive']) for b in r['surfaces'] if b['count'] and b['semantic'] in (51004,51005))
    results=[];identities={}
    for report,case in cases:
        r=json.loads(report.read_bytes());p=next((case/'native-traces/accepted-geometry').glob('step-64.mrvpack'))
        meta=json.loads(p.with_suffix('.receipt.json').read_bytes())
        if r['accepted_pack_sha256']!=meta['pack_file_sha256'] or audit.sha(p)!=meta['pack_file_sha256']:raise ValueError('Unmatched native geometry')
        pose={q['body_index']:q for q in meta['accepted_body_poses']}
        v,f=read_skin(p,meta);skin=np.ascontiguousarray(v,dtype='d');f=np.ascontiguousarray(f,dtype='i8')
        samples=[];weights=[];groups=[]
        with p.open('rb')as st,mmap.mmap(st.fileno(),0,access=mmap.ACCESS_READ)as raw:
            for (sem,sid),prim,vert,tri in audit.primitives(raw,pose):
                k=(sem,sid,prim)
                if k not in banks:continue
                key='/'.join(map(str,k))
                # Sample the same source-face strata for all cases with a deterministic midpoint per stratum.
                bounds=np.linspace(0,len(tri),min(len(tri),512)+1,dtype=int)
                selected=(bounds[:-1]+bounds[1:]-1)//2
                if key in identities and not np.array_equal(identities[key],tri[selected]):raise ValueError('Changed target face identity')
                identities[key]=tri[selected].copy()
                points=vert[tri].astype(float);a=np.linalg.norm(np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0]),axis=1)/2
                samples.append(points[selected].mean(axis=1));weights.append(np.add.reduceat(a,bounds[:-1]))
                groups.extend([key]*len(selected))
        q=np.ascontiguousarray(np.vstack(samples));area=np.concatenate(weights);groups=np.array(groups)
        winding=np.abs(igl.fast_winding_number(skin,f,q));tree=igl.AABB();tree.init(skin,f)
        d=np.sqrt(np.maximum(0,tree.squared_distance(skin,f,q)[0]))
        result={}
        for key in sorted(set(groups)):
            idx=groups==key;ex=idx & (winding<.2);amb=idx & (winding>=.2) & (winding<=.8)
            result[key]=dict(samples=int(idx.sum()),sampled_outside=int(ex.sum()),ambiguous=int(amb.sum()),
                estimated_outside_area_mm2=float(area[ex].sum()*1e6),ambiguous_area_mm2=float(area[amb].sum()*1e6),
                max_sampled_outside_mm=float(d[ex].max(initial=0)*1e3))
        results.append(dict(pack=meta['pack_file_sha256'],per_bank=result))
    out.write_text(json.dumps(results,indent=2)+'\n')
    for r in results:
        for sem in ('51004','51005'):
            sel=[v for k,v in r['per_bank'].items()if k.startswith(sem+'/')]
            print(sem,len(sel),sum(x['estimated_outside_area_mm2']for x in sel),max(x['max_sampled_outside_mm']for x in sel),sum(x['ambiguous']for x in sel),flush=True)
if __name__=='__main__':run([(Path(sys.argv[i]),Path(sys.argv[i+1])) for i in range(1,len(sys.argv)-1,2)],Path(sys.argv[-1]))
