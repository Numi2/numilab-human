"""Compare emitted skin vertices with the points where the native body applies bed force."""
import json,sys,runpy,struct,subprocess,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from material_context import sha
def check(case,opening,reader,reference):
    sys.path.insert(0,str(reader.parent/'src'));api=runpy.run_path(str(reader));clearance=api['clearance']
    state=json.loads((case/'terminal.receipt.json').read_text());control=json.loads((opening/'terminal.receipt.json').read_text())
    if state['base_pack_file_sha256']!=control['base_pack_file_sha256']:raise ValueError('opening pack mismatch')
    table=np.genfromtxt(case/'registered-bed-columns.csv',names=True,delimiter=',')
    meta=state['skin_source_mapping'];n=meta['source_vertex_count'];first=meta['pack_first_vertex']
    def get(k,dt):
        d=meta[k];f=reference/Path(d['path']).name
        if sha(f)!=d['sha256']:raise ValueError('source map changed')
        return np.memmap(f,mode='r',dtype=dt)
    maps=get('vertex_map',np.dtype([('i','<u4',(4,)),('p','<f4',(4,)),('q','<f4',(4,))]))
    w=get('influences',np.dtype([('p','<f4',(4,)),('normal','<f4',(4,)),('body','<u4',(4,))]))
    vids=table['vertex'].astype('i8')
    if len(maps)!=n or meta['influence_layout']!='vertex-major' or len(np.unique(vids))!=len(vids) or min(vids)<0 or max(vids)>=n:raise ValueError('unknown skin layout')
    areas=np.zeros(n);areas[vids]=table['area_m2'].astype('f4')
    spec=json.loads((case.parent/'foundation.json').read_text())
    results=[]
    for initial in (True,False):
        poses=control['initial_anatomical_registration']['body_poses'] if initial else state['accepted_body_poses']
        poses=sorted(poses,key=lambda p:p['body_index'])
        if [p['body_index'] for p in poses]!=list(range(state['accepted_body_pose_count'])):raise ValueError('incomplete pose')
        ro=Rotation.from_quat([p['quaternion_xyzw'] for p in poses]).as_matrix();tr=np.array([p['position_m'] for p in poses])
        raw=subprocess.check_output(['zstd','-qdc',case/('initial.mrvpack.zst' if initial else 'accepted.mrvpack.zst')])
        if hashlib.sha256(raw).hexdigest()!=state['base_pack_file_sha256' if initial else 'pack_file_sha256']:raise ValueError('wrong native pack')
        h=clearance._PACK_HEADER;dire=clearance._PACK_DIRECTORY
        sections=dict((r[0],r) for r in (dire.unpack_from(raw,h.size+i*dire.size) for i in range(h.unpack_from(raw)[2])))
        vertex,ind,prim=map(sections.get,(2,3,4));xyz=np.ndarray((vertex[4],20),dtype='<f4',buffer=raw,offset=vertex[2])[first:first+n,:3]
        used=None
        for j in range(prim[4]):
            ix,ct,_,instance,semantic,_,_,stable=struct.unpack_from('<8I',raw,prim[2]+j*prim[5])
            if (semantic,stable)==api['SKIN']:
                indices=np.frombuffer(raw,dtype='<u4',count=ct,offset=ind[2]+4*ix).astype('i8')-first
                if np.min(indices)<0 or np.max(indices)>=n:raise ValueError('wrong skin offset')
                used=np.unique(indices)
                section=sections[5];inst=struct.unpack_from('<8f12I',raw,section[2]+instance*section[5])
                if inst[10]!=0:raise ValueError('unsupported rigid owner for whole skin')
                xyz=api['world_positions'](xyz,inst)
                break
        if used is None:raise ValueError('skin absent')
        # The bed uses rigid donor transforms and the same immutable LBS point weights.
        ans=np.empty((n,3))
        for i in range(0,n,512):
            m=maps['i'][i:i+512,:2].astype('i8');start=m[:,0]-meta['influence_first_index'];counts=m[:,1]
            b=np.r_[0,np.cumsum(counts)[:-1]]
            if min(counts)==0:raise ValueError('unbound source vertex')
            ix=np.arange(sum(counts))+np.repeat(start-b,counts)
            if ix.min()<0 or ix.max()>=len(w):raise ValueError('invalid owner range')
            inf=w[ix];owners=inf['body'][:,0];p=inf['p'].astype(float)
            world=np.einsum('nij,nj->ni',ro[owners],p[:,:3])+tr[owners]
            ans[i:i+len(m)]=np.add.reduceat(world*p[:,3,None],b)
        err=xyz.astype(float)-ans
        active=(areas>0)&(ans[:,2]<spec['free_surface_z_m'])
        vis=np.zeros(n,bool);vis[used]=True
        if initial and abs(ans[vids,2]-table['z_m']).max()>5e-6:raise ValueError('wrong opening pose')
        # Recompute the compressed mattress face with the source traction laws.
        # This calculation reads the exported geometry, never the renderer output buffers.
        interface = {}
        if not initial:
            from scipy.optimize import brentq
            cfg = json.loads((case/'launch.json').read_text())
            path, = [Path(k) for k in cfg['pins'] if k.endswith('/curves.json')]
            if sha(path) != spec['material']:
                raise ValueError('different tissue material in boundary comparison')
            mats = json.loads(path.read_text())['material']['materials']
            pp = np.unique(np.r_[np.linspace(0,spec['pressure_limit_pa'],1025),
                                 [x for x in spec['pressure_pa'] if x <= spec['pressure_limit_pa']]])
            strain = np.array([[1-brentq(lambda l: m['mu_pa']*(1/l-l)-m['log_volume_coefficient_pa']*np.log(l)/l-pressure,
                                  .66,1.) for m in mats] for pressure in pp])
            sag = spec['foam_height_m']*np.interp(pp,spec['pressure_pa'],spec['compression_fraction'])
            rows = active[vids] & vis[vids]
            v = vids[rows]
            depths = spec['free_surface_z_m']-ans[v,2]
            skin,fat = table['skin_m'][rows],table['fat_m'][rows]
            lo = np.zeros(len(v),int); hi = np.full(len(v),len(pp)-1,int)
            def closure(k): return skin*strain[k,0]+fat*strain[k,1]+sag[k]
            if np.any(depths>closure(hi)+1e-7):
                raise ValueError('captured geometry exceeds source material range')
            while np.max(hi-lo)>1:
                mid=(hi+lo)//2; below=closure(mid)<depths
                lo=np.where(below,mid,lo); hi=np.where(below,hi,mid)
            t=(depths-closure(lo))/(closure(hi)-closure(lo))
            mattress_z=spec['free_surface_z_m']-(sag[lo]+t*(sag[hi]-sag[lo]))
            gap=xyz[v,2]-mattress_z
            witness=int(np.argmax(np.abs(gap)))
            interface=dict(vertices=len(v),max_absolute_gap_m=float(abs(gap[witness])),
                           min_gap_m=float(gap.min()),max_gap_m=float(gap.max()),worst_vertex=int(v[witness]),
                           gaps_outside_1mm=int(np.sum(abs(gap)>.001)))
        def metrics(mask):
            lens=np.linalg.norm(err[mask],axis=1)
            return dict(vertices=int(sum(mask)),max_m=float(lens.max()),rms_m=float(np.sqrt(np.mean(lens*lens))),maximum_height_error_m=float(np.abs(err[mask,2]).max()))
        results.append(dict(initial=initial,loaded_interface=interface,visible=metrics(vis),loaded_visible=metrics(active&vis),
            source_failure_vertex=17764,failure_vertex_visible=bool(vis[17764]),failure_vertex_error_xyz_m=err[17764].tolist(),loaded_unreferenced=int(sum(active&~vis))))
    (case/'skin-load-geometry.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':check(*(Path(x).resolve() for x in sys.argv[1:]))
