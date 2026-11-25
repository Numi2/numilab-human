"""Measure the executed APL force paths against anatomically corresponding native muscle surfaces."""
import json,mmap,struct,sys
from pathlib import Path
import igl,numpy as np
from scipy.spatial.transform import Rotation
from bindings import audit,read_skin
from measure_route_motion import segments_in_solid
from compile_registered_boundary import sha
from fit_route_binding import asset

def analysis(case,out):
    cfg=json.loads((case/'launch.json').read_bytes())
    m=Path(cfg['argv'][2]);raw=m.read_bytes()
    if sha(m)!=cfg['pins'][str(m)]:raise ValueError('Unmatched muscle mechanics')
    header=struct.unpack_from('<8s9I32s',raw)
    r0=76+header[4]*16+header[5]*64;mt=r0+header[6]*16
    rec=next((case/'native-traces/accepted-geometry').glob('step-64.receipt.json'));meta=json.loads(rec.read_bytes())
    pack=Path(meta['accepted_pack_path'])
    if sha(pack)!=meta['pack_file_sha256']:raise ValueError('Foreign frame')
    poses={b['body_index']:b for b in meta['accepted_body_poses']}
    verts,faces=read_skin(pack,meta);v=np.ascontiguousarray(verts,dtype='d')*1000;f=np.ascontiguousarray(faces,dtype='i8')
    shell=igl.AABB();shell.init(v,f); rows={}
    with pack.open('rb')as st,mmap.mmap(st.fileno(),0,access=mmap.ACCESS_READ)as rawpack:
        for (sem,sid),prim,x,y in audit.primitives(rawpack,poses):
            if sem!=51005 or sid not in [147,148]:continue
            actuator={147:259,148:322}[sid];_,start,n,_=struct.unpack_from('<4I',raw,mt+164*actuator)
            nodes=[];donors=[]
            for k in range(start,start+n):
                kind,site,_,_=struct.unpack_from('<4I',raw,r0+16*k)
                if kind!=1:raise ValueError('Wrapped muscle requires the actual routed solver')
                b,*point=struct.unpack_from('<I3f',raw,76+16*site);q=poses[b];donors.append(b)
                nodes.append((Rotation.from_quat(q['quaternion_xyzw']).apply(point)+np.array(q['position_m']))*1000)
            nodes=np.array(nodes);mesh,inv=np.unique(x.astype(float)*1000,axis=0,return_inverse=True);tri=np.ascontiguousarray(inv[y],dtype='i8')
            muscle=igl.AABB();muscle.init(mesh,tri)
            Q=np.ascontiguousarray(mesh[tri].mean(axis=1));w=np.abs(igl.fast_winding_number(v,f,Q));d2,nearest,surface=shell.squared_distance(v,f,Q)
            bad=np.flatnonzero(w<.2);i=bad[np.argmax(d2[bad])]
            t=Q[i]-nodes[:-1];edge=np.diff(nodes,axis=0);a=np.clip((t*edge).sum(axis=1)/(edge*edge).sum(axis=1),0,1)
            nearest_route=np.linalg.norm(t-a[:,None]*edge,axis=1);s=int(np.argmin(nearest_route))
            intervals=[]
            for j in range(n-1):
                pts=nodes[j:j+2];length=np.linalg.norm(pts[1]-pts[0]);inside=sum(a[3]for a in segments_in_solid(shell,v,f,pts));overlap=sum(a[3]for a in segments_in_solid(muscle,mesh,tri,pts))
                intervals.append(dict(length_mm=length,skin_outside_mm=max(0,length-inside),within_muscle_mm=overlap))
            rows[sid]=dict(actuator=actuator,owners=donors,segments=intervals,exterior_source_triangle=int(i),skin_triangle=int(nearest[i]),world_mm=Q[i].tolist(),skin_distance_mm=np.sqrt(d2[i]),nearest_route_segment=s,nearest_route_fraction=a[s],distance_to_force_line_mm=nearest_route[s])
    out.write_text(json.dumps(dict(pack=sha(pack),rows=rows),indent=2)+'\n')
    for sid,row in rows.items():print(sid,'exterior depth mm',row['skin_distance_mm'],'route distance mm',row['distance_to_force_line_mm'],'outside along route mm',sum(a['skin_outside_mm']for a in row['segments']),'inside active muscle surface mm',sum(a['within_muscle_mm']for a in row['segments']),flush=True)

if __name__=='__main__':analysis(*map(Path,sys.argv[1:]))
