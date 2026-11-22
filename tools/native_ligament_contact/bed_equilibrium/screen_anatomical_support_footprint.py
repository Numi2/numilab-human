"""Screen rigid skeleton clearance and available posterior skin footprint in the admitted joint pose."""
import json,sys,runpy,subprocess,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from evaluate_registered_bed_transfer import digest
def clipped_area(points,z):
    poly=[]
    for a,b in zip(points,np.roll(points,-1,axis=0)):
        below=a[2]<=z
        if below:poly.append(a[:2])
        if below!=(b[2]<=z):
            u=(z-a[2])/(b[2]-a[2]);poly.append(a[:2]+u*(b[:2]-a[:2]))
    if len(poly)<3:return 0.
    uv=np.array(poly)
    return abs(np.dot(uv[:,0],np.roll(uv[:,1],-1))-np.dot(uv[:,1],np.roll(uv[:,0],-1)))*.5
def main(full,reader,moment,out):
    job=json.loads((full/'launch.json').read_text());env=job['environment'];prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_SURFACE_O_'
    n=Path(env[prefix+'NODES_CSV']);f=Path(env[prefix+'FACES_CSV']);assert digest(n)==env[prefix+'NODES_SHA256'] and digest(f)==env[prefix+'FACES_SHA256']
    x=np.loadtxt(n,skiprows=1,delimiter=',',usecols=(2,3,4)).astype('f4').astype(float)
    faces=np.loadtxt(f,skiprows=1,delimiter=',',usecols=(1,2,3),dtype=int);tri=x[faces]
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);proj=np.maximum(0.,-normal[:,2]/2)
    # The O facets are stored with a verified material orientation. Select posterior-facing facets.
    rc=full/'native/accepted-geometry/step-16.receipt.json';s=json.loads(rc.read_text())
    pack=full/'terminal.mrvpack.zst';ad=json.loads((full/'admission.json').read_text())
    assert digest(pack)==ad['terminal_archive_sha256'] and s['accepted_step']==16
    raw=subprocess.check_output(['/opt/homebrew/bin/zstd','-qdc',str(pack)]);assert hashlib.sha256(raw).hexdigest()==s['pack_file_sha256']
    sys.path.insert(0,str(reader.parent/'src'));api=runpy.run_path(str(reader));poses={b['body_index']:b for b in s['initial_anatomical_registration']['body_poses']}
    bones=[]
    for key,i,pt,fac in api['primitives'](raw,poses):
        if key[0]==51004:bones.append((float(pt[:,2].min()),int(key[1]),int(i)))
    minz,bone,prim=min(bones);assert len(bones)>100
    com=np.array(json.loads(moment.read_text())['centre_of_accounted_mass_xy_m'])
    rows=[]
    for mm in (1.953125,3,4,6,8,10,11.71875,12,16,20,25,30,40):
        z=mm*.001;active=x[:,2]<=z;cut=(tri[:,:,2].min(1)<=z)&(tri[:,:,2].max(1)>z)&(proj>0);full_cut=tri[:,:,2].max(1)<=z
        area=proj[full_cut].sum()+sum(clipped_area(t,z) for t in tri[cut])
        hull=ConvexHull(x[active,:2]).equations;inside=(hull[:,:2]@com+hull[:,2]).max()<=0
        rows.append(dict(descent_mm=mm,source_nodes=int(active.sum()),posterior_projected_area_cm2=float(area*1e4),
                         lowest_bone_gap_mm=float(1000*(minz-z)),projected_com_in_hull=bool(inside)))
    d=dict(source_receipt_sha256=digest(rc),source_pack_sha256=s['pack_file_sha256'],source_outer_sha256=digest(n),
           minimum_rigid_bone_height_m=minz,bone_id=bone,primitive=prim,rigid_banks=len(bones),rows=rows,
           scope='Rigid registered joint pose and undeformed outer skin. Positive clearance and planform area alone cannot establish pressure, equilibrium or lack of soft-organ crossings.')
    out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['rows']))
if __name__=='__main__':main(*(Path(v).resolve() for v in sys.argv[1:]))
