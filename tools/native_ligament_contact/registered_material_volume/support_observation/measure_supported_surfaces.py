"""Compare the accepted exterior graphics surface and passive tissue banks with actual bed coordinates."""
import csv,json,mmap,runpy,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from trace_loaded_columns import hits
from compile_connected_joint import digest
def run(audit,case,rays,material,output):
    rr=json.loads(rays.read_bytes());n=np.array(rr['bed_normal']);geo=case/'native-traces/accepted-geometry/step-1.receipt.json';f=json.loads(geo.read_bytes());pack=Path(f['accepted_pack_path']);cfg=json.loads((case/'launch.json').read_bytes())
    if digest(geo)!=rr['pose'] or digest(pack)!=f['pack_file_sha256']:raise ValueError('Different accepted geometry')
    api=runpy.run_path(str(audit));primitives=api['primitives'];poses={p['body_index']:p for p in f['accepted_body_poses']}
    x=np.loadtxt(case/'physical-accepted.node',skiprows=1,usecols=(1,2,3))
    v=material/'witness-material/volume';w=np.load(v/'material-walls.npz');outer=np.unique(w['O'])
    axis=np.cross([0,1,0],n);axis/=np.linalg.norm(axis);frame=np.stack((axis,np.cross(n,axis),n)).T
    ids=[r['node']for r in rr['loaded_outer_nodes']];queries=x[ids]@frame
    banks=[[]for i in ids]; report={}
    packet=Path(cfg['argv'][cfg['argv'].index('--support-contact-payload')+1])
    import struct
    h=np.array(struct.unpack_from('<3f',packet.read_bytes(),56));assert digest(packet)==cfg['pins'][str(packet)]
    with pack.open('rb')as file,mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ)as m:
        for key,idx,px,faces in primitives(m,poses):
            if key==api['SKIN']:
                nearest,k=cKDTree(x[outer]).query(px,k=1)
                report.update(rendered_active_vertices=len(px),physical_outer_nodes=len(outer),matched_unique_physical_nodes=len(np.unique(k)),render_to_physical_max_m=float(nearest.max()),render_to_physical_rms_m=float(np.sqrt(np.mean(nearest**2))),rendered_fixed_plane_min_m=float(np.min((px-h)@n)),physical_fixed_plane_min_m=float(np.min((x[outer]-h)@n)))
            if key[0]!=51005:continue
            coord=px.astype(float)@frame
            if np.any(queries[:,:2].min(0)>coord[:,:2].max(0)) or np.any(queries[:,:2].max(0)<coord[:,:2].min(0)):continue
            for j,inter in enumerate(hits(coord,faces,key[1],queries)):
                banks[j].extend(inter)
    report['trace_reported_gap_m']=float(list(csv.DictReader((case/'native-traces/resting-surface-audit.csv').open()))[-1]['min_skin_bed_gap_m'])
    report['physical_skin_to_muscle_rays']=[dict(node=i,intersections=[dict(depth_m=d,stable=int(key))for d,key in sorted(group)[:12]])for i,group in zip(ids,banks)]
    report.update(accepted_pack=digest(pack),physical_state=digest(case/'physical-accepted.node'),support=digest(packet));output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':run(*(Path(x).resolve()for x in sys.argv[1:]))
