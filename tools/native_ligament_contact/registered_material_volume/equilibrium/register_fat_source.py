"""Rebind a deep-fat surface registration while preserving source vertex and donor ancestry."""
import csv,json,subprocess,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from build_cortical_skin_registration import write_contact_include
def run(control,material,prepared,stage):
    r=json.loads((material/'result.json').read_text());v=material/'volume';s=stage/'witness-material'
    b=s/'binding';tab=b/'tables';app=b/'app';con=b/'contact'
    for p in (tab,app,con):p.mkdir(exist_ok=True,parents=True)
    e=json.loads((control/'launch.json').read_text())['environment']
    pr='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    pin={}
    def source(k):
        f=Path(e[pr+k+'_CSV'])
        if digest(f)!=e[pr+k+'_SHA256']:raise ValueError('Changed '+k)
        return f
    def patch(k,fn):
        file=tab/(k.lower()+'.csv')
        with source(k).open()as f,file.open('w')as o:
            o.write(f.readline())
            for line in f:
                row=line.rstrip('\r\n').split(',')
                o.write((','.join(row)+'\n') if fn(row) else line)
        pin.update({pr+k+'_CSV':str(file),pr+k+'_SHA256':digest(file)})
    fmt=lambda v:[format(float(q),'.17g') for q in v]
    N=set(r['modified_nodes']);X=np.loadtxt(v/'mesh.node',skiprows=1,usecols=(1,2,3));old=np.loadtxt(e[pr+'TETGEN_NODE'],skiprows=1,usecols=(1,2,3))
    if N!=set(np.flatnonzero((old!=X).any(1))):raise ValueError('Source edits differ')
    with source('DONOR_FRAMES').open()as f:donor={int(row['donor']):row for row in csv.DictReader(f)}
    P=np.loadtxt(e['NUMI_H706_FRACTIONAL_SOURCE330_NODES'],skiprows=1,delimiter=',',usecols=(2,3,4,5,6,7))
    T=np.loadtxt(e['NUMI_H706_FRACTIONAL_SOURCE330_TETS'],skiprows=1,delimiter=',',usecols=(3,4,5,6),dtype=int)
    asset=Path(e[pr+'ROLE_D_SOURCE_ASSET'])
    if digest(asset)!=e[pr+'ROLE_D_SOURCE_ASSET_SHA256']:raise ValueError('Changed source role')
    identity=json.loads(asset.read_bytes());points=identity['vertices'];tri=identity['faces']
    source_rows={r['source_vertex_row']:i for i,r in enumerate(points)}
    incident={}
    for f in tri:
        for k,idx in enumerate(f['surface_vertex_rows']):
            incident.setdefault(idx,(f['face_row'],k))
    bysurface={};ancestry={}
    def combined(a):
        n=int(a[0])-1
        if n not in N:return False
        if (a[6],a[8],a[18])!=('D','D','vertex') or any(float(x) for x in a[48:52]):raise ValueError('Expected attached material surface with exact operational reference')
        if np.linalg.norm(np.array(a[9:12],float)-old[n])>1e-14:raise ValueError('Inconsistent control material')
        face=int(a[7]);delta=X[n]-old[n];d=donor[int(a[12])]
        R=np.array([float(d[f'current_R{i}{j}']) for i in range(3) for j in range(3)]).reshape(3,3)
        origin=np.array([float(d['current_'+x])for x in 'xyz'])
        a[9:12]=fmt(X[n]);a[14:17]=fmt(np.linalg.solve(R,X[n]-origin))
        vertex=int(a[19]);idx=source_rows[vertex];base=points[idx];sid,corner=incident[idx]
        corners=tri[sid]['surface_vertex_rows'];bc=np.eye(3)[corner]
        base_xyz=bc@np.array([points[k]['current_world_m']for k in corners])
        if points[corners[corner]]['source_vertex_row']!=vertex or base['registered_frame_owner_body']!=int(a[12]):raise ValueError('Unmatched source vertex in original face')
        # Earlier registration edits can already separate a source and its physical vertex.
        # Preserve that measured offset along with this edit, in the original face frame.
        inherited=old[n]-base_xyz
        if np.max(abs(inherited))>np.max(abs(np.spacing(old[n].astype('f4'))))*2:raise ValueError('Unexplained previous source offset')
        a[18:27]=['face','-1',str(sid),*fmt(bc),*fmt(X[n]-base_xyz)]
        ancestry[face]=dict(vertex=vertex,face=sid,corner=corner,material_node=n,inherited_source_offset_m=inherited.tolist())
        tet=P[T[int(a[29])]]
        uvw=np.linalg.solve((tet[1:,3:]-tet[0,3:]).T,X[n]-tet[0,3:]);beta=np.r_[1-uvw.sum(),uvw]
        if min(beta)<=0:raise ValueError('Source parent membership changed')
        residual=X[n]-beta@tet[:,:3]
        a[30:34]=fmt(beta);a[34:38]=fmt([*residual,np.linalg.norm(residual)])
        bysurface[face]=a;return True
    patch('COMBINED_NODE_MAP',combined)
    if len(bysurface)!=len(N):raise ValueError('Missing registered nodes')
    def reg(a):
        if a[0]!='D' or int(a[1]) not in bysurface:return False
        c=bysurface[int(a[1])]
        if (a[3],int(a[15]))!=(c[29],ancestry[int(a[1])]['vertex']):raise ValueError('Source identity changed')
        a[4:13]=c[30:39];a[15:23]=c[19:27];return True
    patch('NODE_REGISTRATION',reg)
    vertices=np.loadtxt(source('SURFACE_D_NODES'),skiprows=1,delimiter=',',usecols=(2,3,4,5,6,7));xyz=vertices[:,:3].copy()
    for i,c in bysurface.items():
        n=int(c[0])-1
        if not np.array_equal(xyz[i],old[n]):raise ValueError('Registration coordinate mismatch')
        xyz[i]=X[n]
    faces=np.loadtxt(source('SURFACE_D_FACES'),skiprows=1,delimiter=',',usecols=(1,2,3),dtype=int)
    ring=np.unique(faces[np.isin(faces,list(bysurface)).any(1)])
    def normals(pts):
        dx=np.cross(pts[faces[:,1]]-pts[faces[:,0]],pts[faces[:,2]]-pts[faces[:,0]])
        n=np.zeros_like(xyz)
        for j in range(3):np.add.at(n,faces[:,j],dx)
        return n[ring]
    n0=normals(vertices[:,:3]);n1=normals(xyz);sign=np.sign(np.einsum('ij,ij->i',n0,vertices[ring,3:]));length=np.linalg.norm(n1,axis=1)
    if np.any(sign==0) or np.any(length<=0):raise ValueError('Indeterminate normal')
    normals_by_id=dict(zip(ring,n1*(sign/length)[:,None]))
    def surface(a):
        i=int(a[0])
        if i not in normals_by_id:return False
        a[5:8]=fmt(normals_by_id[i])
        if i in bysurface:
            c=bysurface[i]
            if int(a[30])!=ancestry[i]['vertex']:raise ValueError('Wrong original measured vertex')
            a[2:5]=c[9:12];a[8:11]=c[14:17];a[21:29]=c[19:27];a[30]="-1"
        return True
    patch('SURFACE_D_NODES',surface)
    (tab/'result.json').write_text(json.dumps(dict(environment=pin,registered_nodes=len(N),face_normals_updated=len(ring),ancestry=ancestry,source_registration=digest(asset)),indent=2)+'\n')
    if (b/'build.json').exists():
        existing=json.loads((b/'build.json').read_text())
        if existing['loaded']!=r['loaded'] or existing['reference']!=r['reference'] or digest(app/'numi-human-native')!=existing['app']:raise ValueError('Changed compiled registration')
        return
    base=prepared/'binding';ca=json.loads((base/'contact/result.json').read_text())
    if (ca['material_reference'],ca['loaded'])!=(r['parent_reference'],digest(prepared/'current.node')):raise ValueError('Cortical owner mismatch')
    data=base/'contact/material-contact.npz'
    if np.isin(np.load(data)['facets'][:,5:8],list(N)).any():raise ValueError('Contact attachment moved')
    ca.update(material_reference=r['reference'],loaded=r['loaded'])
    (con/'result.json').write_text(json.dumps(ca,indent=2)+'\n')
    if not (con/data.name).exists():(con/data.name).symlink_to(data.resolve())
    write_contact_include(con,app)
    h='NumiHumanLoadedSkinPoseData.inc';baseapp=base/'app';code=(baseapp/h).read_text()
    for key,oldSHA,newSHA in [('referenceSHA',r['parent_reference'],r['reference']),('nodeSHA',digest(prepared/'current.node'),r['loaded'])]:
        token=key+'[]="'+oldSHA+'"'
        if code.count(token)!=1:raise ValueError('Stale native registration')
        code=code.replace(token,key+'[]="'+newSHA+'"')
    (app/h).write_text(code)
    for p in baseapp.iterdir():
        if p.suffix in ('.mm','.inc','.h','.hpp') and p.name not in (h,'NumiHumanCorticalSkinBinding.inc'):
            subprocess.run(['/bin/cp','-cp',p,app/p.name],check=True)
    tag=json.loads((base/'build.json').read_text());commands=[]
    for cmd in tag['commands']:
        c=[a.replace(str(baseapp)+'/',str(app)+'/') for a in cmd]
        with (b/('compile-'+str(len(commands))+'.log')).open('w')as f:subprocess.run(c,stdout=f,stderr=subprocess.STDOUT,check=True)
        commands.append(c)
    (b/'build.json').write_text(json.dumps(dict(loaded=r['loaded'],reference=r['reference'],app=digest(app/'numi-human-native'),library=tag['library'],commands=commands),indent=2)+'\n')
if __name__=='__main__':run(*(Path(x).resolve()for x in sys.argv[1:]))
