"""Intersect the fixed bed normal through the actual four material walls and articulated bone banks."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from compile_connected_joint import digest
from rebind_ligament_material import banks
def hits(points,triangles,bank,queries):
    pts=points[triangles];center=(pts[:,:,:2].min(1)+pts[:,:,:2].max(1))*.5
    radius=np.max(abs(pts[:,:,:2]-center[:,None,:]),axis=(1,2));result=[[]for _ in queries]
    binids=np.frexp(radius)[1]
    for scale in np.unique(binids):
        ids=np.flatnonzero(binids==scale);tree=cKDTree(center[ids])
        lists=tree.query_ball_point(queries[:,:2],radius[ids].max()+1e-12,p=np.inf)
        for i,subset in enumerate(lists):
            if not subset:continue
            ii=ids[subset];f=pts[ii]
            a=f[:,1,:]-f[:,0,:];b=f[:,2,:]-f[:,0,:];r=queries[i,:2]-f[:,0,:2]
            det=a[:,0]*b[:,1]-a[:,1]*b[:,0];keep=abs(det)>1e-20
            if not keep.any():continue
            ii=ii[keep];a=a[keep];b=b[keep];r=r[keep];det=det[keep]
            u=(r[:,0]*b[:,1]-r[:,1]*b[:,0])/det;v=(a[:,0]*r[:,1]-a[:,1]*r[:,0])/det
            good=(u>=-1e-10)&(v>=-1e-10)&(u+v<=1+1e-10)
            h=pts[ii,0,2]+a[:,2]*u+b[:,2]*v-queries[i,2]
            for j in np.flatnonzero(good&(h>1e-8)):
                result[i].append((float(h[j]),bank))
    return [sorted(row)for row in result]
def run(stage,case,fit,output):
    volume=stage/'witness-material/volume';cfg=json.loads((case/'launch.json').read_bytes())
    f=json.loads((fit/'fit.json').read_bytes());geo=json.loads((case/'native-traces/accepted-geometry/step-1.receipt.json').read_bytes());p=geo['accepted_body_poses']
    src=case/'physical-accepted.node';x=np.loadtxt(src,skiprows=1,usecols=(1,2,3));bone=Path(cfg['argv'][3])
    if cfg['pins'][str(bone)]!=digest(bone) or not geo['accepted_step']==1:raise ValueError('Unmatched source')
    normal=np.array(f['bed_normal']);normal=normal/np.linalg.norm(normal)
    a=np.cross([0,1,0],normal);a/=np.linalg.norm(a);frame=np.stack((a,np.cross(normal,a),normal)).T
    coords=x@frame;wall=np.load(volume/'material-walls.npz');nodes=np.asarray(f['contact_nodes'])
    query=coords[nodes];inter={k:hits(coords,wall[k],k,query)for k in 'IDC'}
    bone_hits=[[]for n in nodes]
    for (body,bank),(tri,v,_) in banks(bone).items():
        pose=p[body];assert pose['body_index']==body
        xy=(Rotation.from_quat(pose['quaternion_xyzw']).apply(v)+pose['position_m']).astype('f4').astype(float)@frame
        if np.any(np.max(query[:,:2],0)<np.min(xy[:,:2],0)) or np.any(np.min(query[:,:2],0)>np.max(xy[:,:2],0)):continue
        for group,new in zip(bone_hits,hits(xy,tri,[int(body),int(bank)],query)):group.extend(new)
    # Identify the interval leaving the adipose material, using the first intersection of its inner face.
    trace=[]
    for i,node in enumerate(nodes):
        h={k:(inter[k][i][0][0] if inter[k][i] else None) for k in 'IDC'}
        b=sorted(bone_hits[i]);first=b[0] if b else [None,None]
        gap=first[0]-h['D'] if first[0] is not None and h['D'] is not None else None
        trace.append(dict(node=int(node),inner_skin_m=h['I'],deep_fat_m=h['D'],cortical_material_m=h['C'],first_bone=first[1],first_bone_m=first[0],gap_from_deep_fat_to_bone_m=gap))
    out=dict(origin_frame=digest(src),pose=digest(case/'native-traces/accepted-geometry/step-1.receipt.json'),bone=digest(bone),bed_normal=normal.tolist(),loaded_outer_nodes=trace)
    output.write_text(json.dumps(out,indent=2)+'\n')
    return trace
if __name__=='__main__':
    import sys
    print(run(*(Path(s).resolve() for s in sys.argv[1:])))
