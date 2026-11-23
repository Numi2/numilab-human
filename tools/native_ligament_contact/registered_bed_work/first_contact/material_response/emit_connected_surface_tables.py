"""Publish the material-bound O/I/D surfaces through the native CSV contracts."""
import csv,json,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from register_material_surfaces import area_normals

def run(h,job,src,mapping,out):
    out.mkdir();pr='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_';cfg=json.loads((job/'posed-app/scene/launch.json').read_bytes())['environment'];z=np.load(src/'registered-boundary.npz')
    original=json.loads((src/'result.json').read_bytes())
    if digest(src/'registered-boundary.npz')!=original['mapping']:raise ValueError('Changed source binding')
    def file(k):
        f=Path(cfg[pr+k+'_CSV'])
        if digest(f)!=cfg[pr+k+'_SHA256']:raise ValueError('Stale table '+k)
        return f
    xyz=np.loadtxt(h/'closed-native/native/mesh.node',skiprows=1,usecols=(1,2,3));rep={};update={}
    def add(k,f):update[k]={'path':str(f),'sha256':digest(f)}
    ndreg=file('NODE_REGISTRATION');state=json.loads((job/'posed-app/scene/registered-initial-bodies.json').read_bytes())
    dirs=np.array([b['current']['local_to_world_row_major_f32'] for b in state['all_bodies']]).reshape(-1,3,3);inv=np.linalg.inv(dirs)
    converters={};creation={};fmt=lambda x:format(float(x),'.17g')
    for role in 'ID':
        nk='SURFACE_'+role+'_NODES';fk='SURFACE_'+role+'_FACES';initial=file(nk);oldf=file(fk)
        nodes=z[role+'_node_joint'];kept=z[role+'_retained_source_node'];faces=z[role+'_face_nodes'];oldface=z[role+'_retained_source_face'];n0=len(kept);f0=len(oldface)
        newpath=out/(role+'-nodes.csv');fpath=out/(role+'-faces.csv')
        with (mapping/(role+'-new.csv')).open() as f:
            fresh=sorted(csv.DictReader(f), key=lambda r:int(r['registered_surface_vertex_row']))
        if [int(r['joint_node_id'])-1 for r in fresh]!=nodes[n0:].tolist():raise ValueError('Volume-to-surface identity mismatch')
        creation[role]=fresh
        olddata=np.loadtxt(initial,delimiter=',',skiprows=1,usecols=(2,3,4,5,6,7))
        old_tri=np.loadtxt(oldf,delimiter=',',skiprows=1,usecols=(1,2,3),dtype='i4')
        initial_normals=olddata[:,3:]
        fsrc=z[role+'_new_node_parent_face'];bc=z[role+'_new_node_barycentric']
        hints=np.vstack((initial_normals[kept],np.einsum('ij,ijk->ik',bc,initial_normals[old_tri[fsrc]])))
        raw_normals=area_normals(xyz[nodes],faces);length=np.linalg.norm(raw_normals,axis=1)
        if np.any(length==0):raise ValueError('Degenerate area normal')
        oriented=np.sign(np.einsum('ij,ij->i',raw_normals,hints))
        if np.any(oriented==0):raise ValueError('Ambiguous orientation')
        normal=raw_normals*(oriented/length)[:,None]
        change=np.zeros(len(nodes),bool);change[faces[f0:]]=True
        removed=np.ones(len(old_tri),bool);removed[oldface]=False
        rr=np.full(len(olddata),-1,'i4');rr[kept]=np.arange(n0);affected=rr[old_tri[removed]];change[affected[affected>=0]]=True
        converters[role]=rr
        with initial.open() as inp,newpath.open('x') as dest:
            reader=csv.DictReader(inp);w=csv.DictWriter(dest,reader.fieldnames,lineterminator='\n');w.writeheader()
            for r in reader:
                j=int(r['vertex']);i=rr[j]
                if i<0:continue
                if not np.array_equal(olddata[j,:3],xyz[nodes[i]]):raise ValueError('Retained world point drift')
                r['vertex']=i;r['d37_parent_node_row']=i
                if change[i]:
                    for axis,v in zip('xyz',normal[i]):r['n'+axis]=fmt(v)
                w.writerow(r)
            for v in fresh:
                i=int(v['registered_surface_vertex_row']);donor=int(v['donor_body'])
                r=dict(vertex=i,donor=donor,source_domain=v['source_domain'],d37_parent_node_row=-1,d37_parent_surface_face_row=v['source_face_row'],source_vertex_row=-1,frame_owner_method=v['frame_owner_method'])
                for a,axis in enumerate('xyz'):
                    r['w'+axis]=v['current_'+axis+'_m'];r['n'+axis]=fmt(normal[i,a]);r['l'+axis]=v['donor_local_'+axis+'_m'];r['beta'+str(a)]=v['source_bary'+str(a)];r['source_residual_'+axis+'_m']=v['source_residual_'+axis+'_m']
                    for b in range(3):r['m'+str(a)+str(b)]=fmt(inv[donor,a,b])
                w.writerow(r)
        material=15 if role=='I' else 16
        middle=z[role+'_new_face_sample_source'][:,3];parent_cells=np.loadtxt(oldf,delimiter=',',skiprows=1,usecols=(5,),dtype='i4')
        old_rows=np.full(len(old_tri),-1,'i4');old_rows[oldface]=np.arange(f0)
        with oldf.open() as inp,fpath.open('x') as dest:
            rd=csv.DictReader(inp);w=csv.DictWriter(dest,rd.fieldnames,lineterminator='\n');w.writeheader()
            for r in rd:
                i=old_rows[int(r['face'])]
                if i<0:continue
                r['face']=i
                for col,j in enumerate(faces[i]):r['v'+str(col)]=j
                w.writerow(r)
            for i,j in enumerate(middle,f0):
                tri=faces[i];w.writerow(dict(face=i,v0=tri[0],v1=tri[1],v2=tri[2],d37_parent_surface_face_row=j,d37_parent_tet_row=parent_cells[j],material_index=material,source_role='cavity-retriangulated-historical-surface-centroid'))
        add(nk,newpath);add(fk,fpath);rep[role]={'nodes':len(nodes),'faces':len(faces),'changed_normals':int(change.sum()),'new_embedded_nodes':len(fresh)}
        # One barycentric stencil transfers intensive material fields. Existing rows
        # keep their exact text representation, including full authored weight precision.
        for suffix,fields in [('FULL86',86),('CONTACT_THICKNESS',2)]:
            key='ROLE_'+role+'_'+suffix;source=file(key);dst=out/(role+'-'+suffix+'.csv');wanted=np.unique(old_tri[fsrc])
            subset=set(wanted.tolist());field={}
            with source.open() as inp,dst.open('x') as dest:
                dest.write(inp.readline())
                for j,line in enumerate(inp):
                    a=line.rstrip().split(',')
                    if int(a[0])!=j or len(a)!=fields+1:raise ValueError('Unexpected material field row')
                    if j in subset:field[j]=np.array(a[1:],float)
                    if rr[j]>=0:dest.write(str(rr[j])+','+','.join(a[1:])+'\n')
                if j+1!=len(rr):raise ValueError('Incomplete '+key)
                for i,(weights,tri) in enumerate(zip(bc,old_tri[fsrc]),n0):
                    value=weights@np.stack([field[v] for v in tri])
                    if np.any(value<0) or not np.isfinite(value).all() or (fields==86 and abs(value.sum()-1)>1e-5):raise ValueError('Unphysical interpolation for '+key)
                    dest.write(str(i)+','+','.join(map(fmt,value))+'\n')
            add(key,dst)
    path=out/'node-parent-registration.csv'
    with ndreg.open() as inp,path.open('x') as dest:
        rd=csv.DictReader(inp);w=csv.DictWriter(dest,rd.fieldnames,lineterminator='\n');w.writeheader()
        for r in rd:
            k=r['layer']
            if k in converters:
                i=converters[k][int(r['surface_vertex_row'])]
                if i<0:continue
                r['surface_vertex_row']=i
            w.writerow(r)
        for role,rows in creation.items():
            for r in rows:
                rr=dict(layer=role,surface_vertex_row=r['registered_surface_vertex_row'])
                for key in rd.fieldnames:
                    if key in rr:continue
                    rr[key]=r[key.replace('parent_residual_', 'parent_current_residual_')]
                w.writerow(rr)
    add('NODE_REGISTRATION',path)
    add('COMBINED_NODE_MAP',mapping/'combined-node-map.csv')
    (out/'result.json').write_text(json.dumps(dict(inputs=original['mapping'],layers=rep,files=update),indent=2)+'\n')
if __name__=='__main__':run(*(Path(x).resolve() for x in sys.argv[1:]))
