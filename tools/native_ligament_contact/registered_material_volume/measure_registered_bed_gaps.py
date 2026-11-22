"""Measure the actual outer skin area within given normal distances of the native plane."""
import csv,json,struct,sys
from pathlib import Path
import numpy as np
from emit_skin_source_registration import sha

launch,accepted,output=map(Path,sys.argv[1:])
c=json.loads(launch.read_text()); args=c['argv']; env=c['environment']
base='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
def source(key):
    path=Path(env[base+key+'_CSV'])
    if sha(path)!=env[base+key+'_SHA256']:raise ValueError('Material surface pin changed')
    return path
plane=Path(args[args.index('--support-contact-payload')+1]);raw=plane.read_bytes()
manifest=plane.parent/'resting-supine-scene.manifest.json'
support=json.loads(manifest.read_text())['outputs']['support_contact']
assert Path(support['path']).resolve()==plane.resolve() and support['bytes']==len(raw)
assert sha(plane)==support['sha256'] and raw[:8]==b'NHCNT1\0\0'
abi,_,rows,_=struct.unpack_from('<4I',raw,8)
assert abi==1 and len(raw)==84+48*rows
point,normal=np.array(struct.unpack_from('<6f',raw,56)).reshape(2,3)
assert abs(np.linalg.norm(normal)-1)<1e-6
nodes=source('SURFACE_O_NODES');faces=source('SURFACE_O_FACES')
ref=np.loadtxt(nodes,delimiter=',',skiprows=1,usecols=(2,3,4))
tri=np.loadtxt(faces,delimiter=',',skiprows=1,usecols=(1,2,3),dtype='i4')
mp=source('COMBINED_NODE_MAP');idx=np.full(len(ref),-1,dtype='i4')
with mp.open() as f:
    for r in csv.DictReader(f):
        if r['registered_surface_layer']=='O':
            i=int(r['registered_surface_vertex_row']);assert idx[i]==-1
            idx[i]=int(r['joint_node_id'])-1
assert idx.min()>=0 and len(np.unique(idx))==len(ref)
def physical_nodes(path):
    with path.open() as f:
        n,dim,attr,b=map(int,next(f).split());arr=np.loadtxt(f)
    assert arr.shape==(n,4) and (dim,attr,b)==(3,0,0)
    assert np.array_equal(arr[:,0],np.arange(1,n+1)) and np.isfinite(arr).all()
    return arr[idx,1:]
paired=accepted.with_name(accepted.name.replace('-accepted.node','-reference.node'))
assert paired!=accepted and np.array_equal(physical_nodes(paired),ref)
def area(x):
    z=(x-point)@normal; v=x[tri]
    a=np.linalg.norm(np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]),axis=1)*.5
    lo,mid,hi=np.sort(z[tri],axis=1).T
    levels=[]
    for h in (0.,1e-6,1e-5,1e-4,5e-4,.001,.002,.005):
        below=(hi<=h);lower=(lo<h)&(h<mid);upper=(mid<=h)&(h<hi)
        frac=below.astype('f8')
        frac[lower]=(h-lo[lower])**2/((mid[lower]-lo[lower])*(hi[lower]-lo[lower]))
        frac[upper]=1-(hi[upper]-h)**2/((hi[upper]-mid[upper])*(hi[upper]-lo[upper]))
        assert np.isfinite(frac).all() and np.all((frac>=-1e-12)&(frac<=1+1e-12))
        levels.append(dict(gap_m=h,vertices=int((z<=h).sum()),skin_area_m2=float(a@frac)))
    return dict(minimum_gap_m=float(z.min()),maximum_gap_m=float(z.max()),levels=levels)
report=dict(scope='Geometric area below each plane offset. The native contact owner computes forces.',support_manifest_sha256=sha(manifest),support_sha256=sha(plane),plane_point_m=point.tolist(),plane_normal=normal.tolist(),native_reference_sha256=sha(paired),native_accepted_sha256=sha(accepted),reference=area(ref),accepted=area(physical_nodes(accepted)))
output.write_text(json.dumps(report,indent=2)+'\n')
