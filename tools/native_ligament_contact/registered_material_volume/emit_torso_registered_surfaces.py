"""Emit the changed physical reference, donor-local coordinates, normals and bonded surface rows."""
import csv,json,sys
from pathlib import Path
import numpy as np
from emit_skin_source_registration import sha

def main():
    launch,reg,proposal,dst=map(Path,sys.argv[1:]);out=dst/'material/registration';out.mkdir(exist_ok=True,parents=True)
    j=json.load(launch.open());env=j['environment'];prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    def inp(k,h):
        p=Path(env[prefix+k]);assert sha(p)==env[prefix+h];return p
    old=reg/'material/registration';s=np.load(proposal);mesh=dst/'mesh'
    with (mesh/'mesh.node').open() as f:n=int(next(f).split()[0]);x=np.loadtxt(f,max_rows=n,usecols=(1,2,3),dtype='f8')
    with inp('TETGEN_NODE','TETGEN_NODE_SHA256').open() as f:n=int(next(f).split()[0]);prior=np.loadtxt(f,max_rows=n,usecols=(1,2,3),dtype='f8')
    with (mesh/'mesh.ele').open() as f:n=int(next(f).split()[0]);t=np.loadtxt(f,max_rows=n,usecols=(1,2,3,4),dtype='i4')-1
    delta=x-prior;moved=np.any(delta!=0,1)
    assert np.array_equal(s['x'],x[s['nodes']]) and np.isin(np.flatnonzero(moved),s['nodes']).all()
    labels=np.loadtxt(mesh/'current-child.labels.csv',delimiter=',',skiprows=1,dtype='i8')
    old_labels=np.loadtxt(inp('CHILD_LABELS','CHILD_LABELS_SHA256'),delimiter=',',skiprows=1,dtype='i8')
    changed_cells=np.flatnonzero(np.any(labels!=old_labels,1));touch_nodes=np.unique(t[changed_cells])
    changed_set=set(map(int,touch_nodes));reclass={}
    if len(touch_nodes):
        for c in np.flatnonzero(np.isin(t,touch_nodes).any(1)):
            for v in t[c]:
                if int(v) in changed_set:reclass.setdefault(int(v),[]).append(int(c))
    transforms={}
    with inp('DONOR_FRAMES_CSV','DONOR_FRAMES_SHA256').open() as f:
        for r in csv.DictReader(f):
            R=np.array([float(r['current_R'+str(i)+str(j)]) for i in range(3) for j in range(3)]).reshape(3,3)
            transforms[int(r['donor'])]=(np.array([float(r['current_'+a]) for a in 'xyz']),np.linalg.inv(R))
    def vec(r,k):return np.array([float(r[k+a+'_m']) for a in 'xyz'])
    def put(r,k,q):
        for a,v in zip('xyz',q):r[k+a+'_m']=format(v,'.17g')
    ids={k:{} for k in 'OID'};surface_res={};changes={};adjusted=0;reassigned=0
    with (old/'combined-node-map.csv').open() as f,(out/'combined-node-map.csv').open('w',newline='') as wf:
        reader=csv.DictReader(f);writer=csv.DictWriter(wf,reader.fieldnames,lineterminator='\n');writer.writeheader()
        for r in reader:
            n=int(r['joint_node_id'])-1;k=r['registered_surface_layer'];row=int(r['registered_surface_vertex_row']) if k in ids else -1
            if k in ids:ids[k][row]=n
            jt=int(r['operational_reference_tet_row'])
            if moved[n] or n in changed_set or np.any(moved[t[jt]]):
                donor=int(r['donor_body']);pos,R=transforms[donor]
                assert np.array_equal(prior[n],vec(r,'current_'))
                q=x[n];put(r,'current_',q);local=R@(q-pos);put(r,'donor_local_',local)
                # Only a face embedding has a residual in source space. A direct vertex ID
                # retains its exact authoring identity. Its physical offset belongs to the
                # current parent registration and the donor-local FEM reference.
                if int(r['source_face_row']) >= 0:
                    put(r,'source_residual_',vec(r,'source_residual_')+delta[n])
                pr=vec(r,'parent_current_residual_')+delta[n];put(r,'parent_current_residual_',pr);r['parent_registration_distance_m']=format(np.linalg.norm(pr),'.17g')
                if np.linalg.norm(pr)>.02:raise ValueError(('Current material parent cap',n))
                if (int(r['geometric_donor_body']),int(r['anatomical_region_id']))!=tuple(labels[jt,2:4]):
                    candidates=reclass[n];gd,gr=int(r['geometric_donor_body']),int(r['anatomical_region_id'])
                    jt=min(candidates,key=lambda c:(labels[c,2]!=gd,labels[c,3]!=gr,c))
                    r['operational_reference_tet_row']=str(jt);r['geometric_donor_body']=str(labels[jt,2]);r['anatomical_region_id']=str(labels[jt,3])
                    bary=(t[jt]==n).astype('f8');adjusted+=1;reassigned+=int((gd,gr)!=tuple(labels[jt,2:4]))
                    assert bary.sum()==1
                    for i,w in enumerate(bary):r['operational_reference_bary'+str(i)]=str(int(w))
                bary=np.array([float(r['operational_reference_bary'+str(i)]) for i in range(4)])
                op=q-bary@x[t[jt]];put(r,'operational_reference_residual_',op);r['operational_reference_distance_m']=format(np.linalg.norm(op),'.17g')
                if np.linalg.norm(op)>.02:raise ValueError(('Operational material cap',n))
                if k in ids:
                    key=(k,row);values=(pr.tolist(),vec(r,'source_residual_').tolist(),local.tolist());assert surface_res.setdefault(key,values)==values
            writer.writerow(r)
    with (old/'node-parent-registration.csv').open() as f,(out/'node-parent-registration.csv').open('w',newline='') as wf:
        reader=csv.DictReader(f);writer=csv.DictWriter(wf,reader.fieldnames,lineterminator='\n');writer.writeheader()
        for r in reader:
            key=(r['layer'],int(r['surface_vertex_row']))
            if key in surface_res:
                pr,sr,_=surface_res[key];put(r,'parent_residual_',pr);put(r,'source_residual_',sr);r['parent_registration_distance_m']=format(np.linalg.norm(pr),'.17g')
            writer.writerow(r)
    for k in 'OID':
        ff=inp('SURFACE_'+k+'_FACES_CSV','SURFACE_'+k+'_FACES_SHA256')
        face=np.loadtxt(ff,delimiter=',',skiprows=1,usecols=(1,2,3),dtype='i4');lm=np.array([ids[k][i] for i in range(len(ids[k]))])
        tri=lm[face];affected=np.unique(tri[moved[tri].any(1)])
        def normals(points):
            e=points[tri[:,1]]-points[tri[:,0]];v=points[tri[:,2]]-points[tri[:,0]]
            nn=np.cross(e,v);norm=np.zeros_like(points)
            for j in range(3):np.add.at(norm,tri[:,j],nn)
            return norm
        oldn=normals(prior);newn=normals(x);active=set(map(int,affected));filename='registered-'+k+'.csv'
        with (old/filename).open() as f,(out/filename).open('w',newline='') as wf:
            reader=csv.DictReader(f);writer=csv.DictWriter(wf,reader.fieldnames,lineterminator='\n');writer.writeheader()
            for r in reader:
                i=int(r['vertex']);n=lm[i]
                if n in active:
                    n0=oldn[n];n1=newn[n];signed=-1 if np.dot(n0,[float(r['n'+a]) for a in 'xyz'])<0 else 1
                    if np.linalg.norm(n1)<1e-20:raise ValueError(('Undefined current area normal',k,i))
                    n1=n1*(signed/np.linalg.norm(n1))
                    for axis,val in zip('xyz',n1):r['n'+axis]=format(val,'.17g')
                if (k,i) in surface_res:
                    _,sr,local=surface_res[k,i]
                    for j,axis in enumerate('xyz'):
                        r['w'+axis]=format(x[n,j],'.17g');r['l'+axis]=format(local[j],'.17g')
                    put(r,'source_residual_',sr)
                writer.writerow(r)
        changes[k]=dict(moved=int(sum(moved[lm])),geometric_normals_recomputed=len(active))
    for name in ('all-body-current-rest.csv','source-domain-pins.tsv','facet-ancestry-current.csv'):
        dest=out/name
        if not dest.exists():dest.hardlink_to(old/name)
    paths={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='receipt.json'}
    report=dict(proposal_sha256=sha(proposal),current_parent_state_sha256=sha(dst/'material/p1/current-paired-volume-state.json'),registered_layers=changes,moved_physical_nodes=int(moved.sum()),reselected_operational_incident_tets=adjusted,changed_numerical_node_region_tuples=reassigned,changed_mechanical_frame_bindings=0,original_facet_ancestry_sha256=sha(old/'facet-ancestry-current.csv'),outputs=paths)
    (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(report['registered_layers'],adjusted,reassigned)
if __name__=='__main__':main()
