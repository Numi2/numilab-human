"""Emit the registered joint-node map for the connected cortical cuts and both physical layers."""
import csv,json,sys
from pathlib import Path
import numpy as np
from bind_material_interior import HistoricalTissue
from compile_connected_joint import digest

def create(h,job,bound,out):
    out.mkdir();doc=json.loads((job/'posed-app/scene/launch.json').read_bytes());env=doc['environment'];pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    def file(k,tag='CSV'):
        f=Path(env[pre+k+'_'+tag])
        if digest(f)!=env[pre+k+'_SHA256']:raise ValueError('Source hash '+k)
        return f
    loc=h/'closed-native/native';x=np.loadtxt(loc/'mesh.node',skiprows=1,usecols=(1,2,3));n=len(x)
    t=np.loadtxt(loc/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype='i4')-1
    labels=np.loadtxt(loc/'current-child.labels.csv',skiprows=1,delimiter=',',usecols=(1,2,3),dtype='i4');mass=np.fromfile(loc/'cell-mass.f64','<f8')
    raw=file('COMBINED_NODE_MAP');hist=HistoricalTissue(env);z=np.load(bound/'registered-boundary.npz')
    proof=json.loads((bound/'result.json').read_bytes())
    if digest(bound/'registered-boundary.npz')!=proof['mapping'] or any(v.get('unresolved_count',0) for v in proof['layers'].values()):raise ValueError('Unresolved surface provenance')
    old_to_new=np.load(h/'scene-registration/node-map.npz')['source_to_emitted'];old=np.full(n,-1,'i4')
    keep=old_to_new>=0;old[old_to_new[keep]]=np.flatnonzero(keep)
    role=np.full(n,-1,'i4');surface=np.full(n,-1,'i4');donors=np.full(n,-1,'i4')
    authored_parent=np.full(n,-1,'i4');source_face=np.full(n,-1,'i4');source_bc=np.zeros((n,3));source_res=np.zeros((n,3))
    for i,k in enumerate('OID'):
        nd=z[k+'_node_joint'];retained=z[k+'_retained_source_node']
        if np.any(role[nd]>=0):raise ValueError('A mesh node has conflicting registered surfaces')
        role[nd]=i;surface[nd]=np.arange(len(nd))
        if len(nd)==len(retained):continue
        sf=z[k+'_new_node_parent_face'];bc=z[k+'_new_node_barycentric']
        registered=np.loadtxt(file('SURFACE_'+k+'_NODES'),delimiter=',',skiprows=1,usecols=(1,30),dtype='i4')
        asset=Path(env[pre+'ROLE_'+k+'_SOURCE_ASSET'])
        if digest(asset)!=env[pre+'ROLE_'+k+'_SOURCE_ASSET_SHA256']:raise ValueError('Immutable source changed')
        dom=json.loads(asset.read_bytes());v=np.array([r['current_world_m'] for r in dom['vertices']]);f=np.array([r['surface_vertex_rows'] for r in dom['faces']])
        if not np.array_equal(registered[:,1],[r['source_vertex_row'] for r in dom['vertices']]) or not np.array_equal(f,np.loadtxt(file('SURFACE_'+k+'_FACES'),skiprows=1,delimiter=',',usecols=(1,2,3),dtype='i4')):raise ValueError('Historical layer order changed')
        added=nd[len(retained):];source_face[added]=sf;source_bc[added]=bc
        authored_parent[added]=np.array([r['parent_tet_row'] for r in dom['faces']],dtype='i4')[sf]
        if np.any(hist.labels[authored_parent[added],0]!=(15 if k=='I' else 16)):raise ValueError('Immutable surface ancestry crosses a material')
        source_res[added]=x[added]-np.einsum('ij,ijk->ik',bc,v[f[sf]])
        donors[added]=registered[f[sf,np.argmax(bc,axis=1)],0];del dom
    actual=json.loads((job/'posed-app/scene/registered-initial-bodies.json').read_bytes());b=actual['all_bodies']
    dirs=np.array([v['current']['local_to_world_row_major_f32'] for v in b]).reshape(-1,3,3);inv=np.linalg.inv(dirs);origin=np.array([v['current']['position_world_m'] for v in b],dtype='f4').astype(float)
    binding={r['body_index']:i for i,r in enumerate(actual['bodies'])}
    tag=np.full((n,3),-1,'i4');old_role=np.full(n,-2,'i4');seen=np.zeros(n,bool);limits={'skin':0,'fat':0};retired_injections=[]
    with raw.open() as inp:
        columns=inp.readline().rstrip().split(',');pos={name:i for i,name in enumerate(columns)}
        for line in inp:
            v=line.rstrip().split(',');j=int(v[0])-1;dest=old_to_new[j];part=v[1];limits[part]=max(limits[part],int(v[2]))
            if dest<0:
                if int(v[42])>=0:retired_injections.append([j,int(v[42])])
                continue
            if seen[dest]:continue
            seen[dest]=True;old_role[dest]={'O':0,'I':1,'D':2,'interior':-1}[v[8]]
            tag[dest]=[hist.labels[int(v[29]),0],int(v[40]),int(v[41])];donors[dest]=int(v[12])
            if not np.array_equal(np.array(v[9:12],float),x[dest]):raise ValueError('Surviving rest point moved')
    if not np.array_equal(seen,old>=0):raise ValueError('Missing historical node')
    if np.any(seen & (role!=old_role)):raise ValueError('Surviving node changed its anatomical layer '+str(np.flatnonzero(seen & (role!=old_role))[:20]))
    # Choose an actual incident corner with the same historical semantic tuple. New
    # nodes use the largest incident material mass grouped by material, donor and region.
    order=np.argsort(t.ravel(),kind='stable').astype('i4');idx=t.ravel()[order];cells=order//4
    off=np.r_[0,np.cumsum(np.bincount(idx,minlength=n))];stop=np.iinfo('i4').max
    chosen=np.full(n,stop,'i4');match=np.all(labels[cells]==tag[idx],axis=1);np.minimum.at(chosen,idx[match],order[match])
    mask=np.zeros(n,'i4');np.bitwise_or.at(mask,idx,1<<labels[cells,0])
    changes=[]
    for node in np.flatnonzero(chosen==stop):
        cs=order[off[node]:off[node+1]];types=labels[cs//4]
        if old[node]>=0:
            cs=cs[types[:,0]==tag[node,0]]
            if not len(cs):raise ValueError('Historical parent tissue absent at surviving node '+str(node))
            same=cs[labels[cs//4,2]==tag[node,2]]
            if len(same):cs=same
        elif role[node]>=0:
            m=16 if role[node]==2 else 15;cs=cs[labels[cs//4,0]==m]
        if not len(cs):raise ValueError('No eligible material owner '+str(node))
        groups,igroup=np.unique(labels[cs//4],axis=0,return_inverse=True)
        g=np.argmax(np.bincount(igroup,weights=mass[cs//4]));chosen[node]=cs[np.flatnonzero(igroup==g)[0]]
        if seen[node]:changes.append([int(node),tag[node].tolist(),groups[g].tolist()])
    if not np.array_equal(t.ravel()[chosen],np.arange(n)):raise ValueError('Wrong physical corner')
    fresh=np.flatnonzero(~seen);geo=labels[chosen//4];donors[donors<0]=geo[donors<0,1]
    local=np.einsum('ijk,ik->ij',inv[donors],x-origin[donors]);frame_error=np.linalg.norm(np.einsum('ijk,ik->ij',dirs[donors],local)+origin[donors]-x,axis=1).max()
    if frame_error>1e-8:raise ValueError('Frame roundtrip')
    # AABB distances bound the exact projection to the closed tetrahedron. Retain every
    # cell that could beat the upper bound or tie the best distance.
    caches={};old_groups=hist.groups.copy()
    def project(node):
        key=tuple(geo[node]);pt=x[node]
        if key not in caches:
            ids=np.array(old_groups[key]);p=hist.points[hist.tet[ids]];caches[key]=(ids,p.min(1),p.max(1),p.mean(1))
        ids,lo,hi,centroid=caches[key];upper=np.linalg.norm(centroid-pt,axis=1).min()+2e-10
        dist=np.linalg.norm(np.maximum(0,np.maximum(lo-pt,pt-hi)),axis=1)
        hist.groups[key]=ids[dist<=upper];return hist.project(pt,key)
    numstr=lambda x:format(float(x),'.17g')
    def base(v,n):
        c=chosen[n]//4;corner=chosen[n]%4
        v[0]=str(n+1);v[9:12]=[numstr(a) for a in x[n]];v[12]=str(donors[n]);v[13]=str(binding.get(int(donors[n]),-1));v[14:17]=[numstr(a) for a in local[n]]
        v[39:42]=list(map(str,[mask[n],geo[n,1],geo[n,2]]));v[43:52]=[str(c)]+[str(int(i==corner)) for i in range(4)]+['0']*4
        if role[n]>=0:
            k='OID'[role[n]];v[6:9]=[k,str(surface[n]),k]
            if k=='I':v[3]=str(surface[n])
        return v
    output=out/'combined-node-map.csv';surface_rows={k:[] for k in 'ID'};projected=[];retained_rows=0;new_count=0
    with output.open('x') as dst,raw.open() as inp:
        dst.write(inp.readline())
        for line in inp:
            row=line.rstrip().split(',');n=old_to_new[int(row[0])-1]
            if n<0:continue
            row=base(row,n);dst.write(','.join(row)+'\n');retained_rows+=1
        for n in fresh:
            if authored_parent[n]>=0:
                # The retained face catalog identifies this exact historical parent.
                # Its source donor/region can differ from the present incident cell.
                q=int(authored_parent[n]);key=tuple(hist.labels[q]);hist.groups[key]=[q]
                q,w,ties=hist.project(x[n],key)
            else:q,w,ties=project(n)
            res=x[n]-w@hist.points[hist.tet[q]];dist=float(np.linalg.norm(res));projected.append(dist)
            v=['']*len(columns);v[3:9]=['-1','','','','-1','interior'];v[17:29]=['derived_volume_interior_no_surface_ancestry','none','-1','-1']+['0']*6+['derived_full_all_bodies_geometric_donor_pose','inferred_current_parent_tet_same_material_donor_region']
            v[29:39]=[str(q)]+list(map(numstr,w))+list(map(numstr,res))+[numstr(dist),str(ties)]
            v[42]='-1';v[52:54]=list(map(str,geo[n,1:]));v=base(v,n)
            if role[n]>=0:
                k='OID'[role[n]]
                if k=='O' or source_face[n]<0:raise ValueError('New boundary embedding missing')
                v[28]='source_parent_tet_reference_registration';v[52:54]=list(map(str,hist.labels[q,1:]))
                v[17:28]=[env[pre+'ROLE_'+k+'_SOURCE_DOMAIN'],'face','-1',str(source_face[n])]+list(map(numstr,source_bc[n]))+list(map(numstr,source_res[n]))+['registered_source_face_barycentric_in_native_donor_frame']
            parts=('skin','fat') if role[n]==1 else (('skin',) if geo[n,0]==15 else ('fat',))
            for a in parts:limits[a]+=1
            for part in parts:
                v[1:3]=[part,str(limits[part])]
                if len(parts)==2:
                    other='fat' if part=='skin' else 'skin';v[4:6]=[other,str(limits[other])]
                dst.write(','.join(v)+'\n');new_count+=1
            if role[n]>=0:surface_rows[k].append(dict(zip(columns,v)))
    for k,v in surface_rows.items():
        with (out/(k+'-new.csv')).open('x') as dst:
            w=csv.DictWriter(dst,columns,lineterminator='\n');w.writeheader();w.writerows(v)
    r=dict(nodes=len(x),source=digest(raw),material_reference=digest(loc/'mesh.node'),elements=digest(loc/'mesh.ele'),emitted=digest(output),source_node_rows=retained_rows,created_rows=new_count,created_nodes=len(fresh),shared_interfaces=int(np.sum(role==1)),retired_historical_parent_nodes=retired_injections,semantic_changes=changes,source_projection_range_m=[min(projected),max(projected)],frame_roundtrip_m=frame_error,body_snapshot=digest(job/'posed-app/scene/registered-initial-bodies.json'),surface_new={k:len(rows) for k,rows in surface_rows.items()})
    (out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print({k:(len(v) if k in ('retired_historical_parent_nodes','semantic_changes') else v) for k,v in r.items()},flush=True)
if __name__=='__main__':create(*(Path(x).resolve() for x in sys.argv[1:]))
