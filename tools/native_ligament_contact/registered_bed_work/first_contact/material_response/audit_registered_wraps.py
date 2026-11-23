"""Evaluate the original MyoSim shoulder routing sphere at the newly compiled native support pose."""
from pathlib import Path
import json,struct,sys
import numpy as np
from scipy.spatial.transform import Rotation
from verify_source_wraps import source_model,path
from rebind_ligament_material import banks
from compile_connected_joint import matrix,digest
from shoulder_inputs import old

def run(control,rigid,bones,material,wrap,out):
    cfg=json.loads((control/'launch.json').read_bytes());records=json.loads((control/'native-initial-bodies.json').read_bytes())['all_bodies'];p=[matrix(q['current']['position_world_m']+q['current']['orientation_world_xyzw']) for q in records]
    raw=(rigid/'source-bound.nhrigid').read_bytes();delta={b:p[b]@matrix(f['parent_translation_base_m']+f['parent_rotation_xyzw'])@np.linalg.inv(p[b]) for b,f in zip([84,34],json.loads((rigid/'frames.json').read_bytes()))}
    for i in range(len(p)-1,-1,-1):
        b=i
        while b:
            root=struct.unpack_from('<I',raw,224+160*157+144*(b-1))[0]
            if root in delta:
                p[i]=delta[root]@p[i];break
            b=root
    def pose(i):return np.r_[p[i][:3,3],Rotation.from_matrix(p[i][:3,:3]).as_quat()]
    bb=next(bones.glob('*.nhbones'));src=banks(bb);lib=json.loads(material.read_bytes());data=json.loads(wrap.read_bytes());tris={}
    if lib['candidate']!=digest(bb):raise ValueError('Mismatched fibers and skeleton')
    for (b,s),(faces,x,_) in src.items():
        t=(x@p[b][:3,:3].T+p[b][:3,3])[faces];tris[s]=t,t.min(1),t.max(1)
    result=[]
    for route in data['routes']:
        side=0 if route['side']=='l' else 1;name='superior_glenohumeral_capsuloligament' if route['name'].startswith('s_gleno') else 'coracohumeral_capsuloligament'
        row=lib['fibers'][name+'/'+str(side)];sites=list(reversed(row['endpoints']));pts=[]
        for site,control_site in zip(sites,route['sites']):
            b=site['body'];s=site['bone'];f,v,_=src[b,s]
            if b!=control_site['body'] or s!=control_site['stable_id']:raise ValueError('Typed bone mismatch')
            pts.append(np.array(site['barycentric'])@v[f[site['triangle']]])
        w=route['source_wrap'];m,d=source_model(*pts,w);line,L,sag=path(m,d,pose(sites[0]['body']),pose(sites[1]['body']),w['center_local_m'])
        hits={}
        for sid,(t,lo,hi) in tris.items():
            for j,(a,b) in enumerate(zip(line,line[1:])):
                mask=((lo<=np.maximum(a,b)+1e-9)&(hi>=np.minimum(a,b)-1e-9)).all(1)
                if not np.any(mask):continue
                ix=np.flatnonzero(mask);tri=t[ix].reshape(-1,3);hit=old.audit.lp.vector_hits(a,b,(tri,np.arange(len(tri)).reshape(-1,3)))
                if hit:hits[sid]=[j,int(ix[hit[0][1]])];break
        result.append(dict(name=name,side=side,source_geometry=w['geom'],native_registered_material_triangles=[s['triangle'] for s in sites],wrap_active=len(line)>2,path_m=L,control_length_m=row['reference_length_m'],arc_error_m=sag,intersected_banks=hits,source_stiffness=route['source_stiffness']))
    out.write_text(json.dumps(dict(source_wrap=digest(wrap),rigid=digest(rigid/'source-bound.nhrigid'),material=digest(material),bones=digest(bb),rows=result),indent=2)+'\n');print([(r['name'],r['side'],r['wrap_active'],list(r['intersected_banks'])) for r in result])
if __name__=='__main__':run(*(Path(k).resolve() for k in sys.argv[1:]))
