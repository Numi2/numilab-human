"""Extract material subdomain PLCs, donor surfaces and their measured native sources."""
import csv,json,runpy,subprocess,sys
from pathlib import Path
import numpy as np
from emit_skin_source_registration import sha
launch,pose,pack,sel,source,impl,owner,report,out=map(Path,sys.argv[1:])
out.mkdir(parents=True,exist_ok=True)
j=json.loads(launch.read_text());env=j['environment'];pfx='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
def get(k,ident):
    p=Path(env[pfx+k]);assert sha(p)==env[pfx+ident];return p
def tet(k,h,cols,dtype):
    with get(k,h).open() as f:
        n=int(next(f).split()[0]);return np.loadtxt(f,usecols=cols,max_rows=n,dtype=dtype,ndmin=2)
x=tet('TETGEN_NODE','TETGEN_NODE_SHA256',(1,2,3),'f8').astype('f4')
cells=np.load(sel);cells_id=cells['cell'];element=tet('TETGEN_ELEMENTS','TETGEN_ELEMENTS_SHA256',(1,2,3,4,5),'i4')[cells_id];t=element[:,:4]-1
tag=element[:,4];nodes=np.unique(t)
def keys(f):
    q=np.sort(f,axis=1).astype(np.int64);return (q[:,0]*len(x)+q[:,1])*len(x)+q[:,2]
def shell(tt):
    face=np.concatenate([tt[:,o] for o in [(1,2,3),(0,3,2),(0,1,3),(0,2,1)]])
    _,row,n=np.unique(keys(face),return_index=True,return_counts=True);assert n.max()==2
    return face[row[n==1]]
sf=[shell(t[tag==r]) for r in [15,16]];sq=np.unique(np.concatenate([keys(f) for f in sf]));lut={}
for row in csv.DictReader(get('FACET_ANCESTRY_CSV','FACET_ANCESTRY_SHA256').open()):
    k=row['face_source_role'];layer={'outer_O':'O','interface_I':'I','inner_I':'I','deep_D':'D'}.get(k)
    ids=sorted(int(row['joint_node'+str(i)])-1 for i in range(3))
    code=(ids[0]*len(x)+ids[1])*len(x)+ids[2]
    if np.searchsorted(sq,code)<len(sq) and sq[np.searchsorted(sq,code)]==code:lut[code]=k
print('roles',set(lut.values()),flush=True)
fixed=set(keys(cells['fixed_boundary']).tolist())
bound={}
for r,f in zip([15,16],sf):
    for v,k in zip(f,keys(f)):
        a=lut.get(k,'');a='C' if k in fixed else ('O' if 'outer_O' in a else 'I' if 'interface' in a or '_I' in a else 'D' if '_D' in a else '')
        if not a:raise ValueError(('Missing material role',r,k,lut.get(k)))
        bound.setdefault(a,{})[k]=v
for a,k in zip('OID',(1,2,3)):
    assert len(bound[a])>0
data=dict(x=x[nodes],nodes=nodes,t=np.searchsorted(nodes,t),role=tag,cell=cells_id,fix=np.searchsorted(nodes,cells['fixed_nodes']))
for a,v in bound.items():data[a]=np.searchsorted(nodes,np.array(list(v.values())))
data['C15']=np.searchsorted(nodes,sf[0][np.isin(keys(sf[0]),list(fixed))])
data['C16']=np.searchsorted(nodes,sf[1][np.isin(keys(sf[1]),list(fixed))])
sys.path.insert(0,str(owner/'tools'));from rebind_human_skin_reference import decode
asset=Path(j['argv'][j['argv'].index('--skin-payload')+1]);data['source_x']=np.loadtxt(source,skiprows=1,delimiter=',',usecols=(2,3,4)).astype('f4')
data['source_f']=decode(asset.read_bytes())['inds'].reshape(-1,3)
ref=json.loads(report.read_text());assert sha(pose)==ref['body_pose_sha256']
raw=subprocess.check_output(['/opt/homebrew/bin/zstd','-qdc',str(pack)])
import hashlib
assert hashlib.sha256(raw).hexdigest()==ref['pack_sha256'];sys.path.insert(0,str(impl.parent/'src'));audit=runpy.run_path(str(impl))
poses={row['body_index']:{'position_m':row['current']['position_world_m'],'quaternion_xyzw':row['current']['orientation_world_xyzw']} for row in json.loads(pose.read_text())['all_bodies']}
for (kind,i),_,v,f in audit['primitives'](raw,poses):
    if kind==51005 and i in (67,68,73,74):data['x'+str(i)]=v;data['f'+str(i)]=f
for i in (67,68,73,74):assert 'x'+str(i) in data
np.savez_compressed(out/'inputs.npz',**data)
print('local facet counts', {a:len(data[a]) for a in ('O','I','D','C15','C16')},flush=True)
