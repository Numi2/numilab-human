"""Bind supported surface samples to contiguous dermis and fat in the registered tetrahedral mesh."""
import csv
import ctypes
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from supported_material_admission import installed_topology


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def native_functions(surface_lib, volume_lib):
    d=np.ctypeslib.ndpointer(dtype="float64",flags="C_CONTIGUOUS")
    i=np.ctypeslib.ndpointer(dtype="int32",flags="C_CONTIGUOUS")
    n=ctypes.c_int
    lower=ctypes.CDLL(str(surface_lib)).bed_boundary_columns
    lower.argtypes=[d,n,i,n,d,n,i,d,d];lower.restype=n
    walk=ctypes.CDLL(str(volume_lib)).trace_registered_columns
    walk.argtypes=[d,d,n,i,i,n,i,i,n,i,d,d,n,i,d];walk.restype=n
    return lower,walk


def prepare(dense,case,opening,surface_lib,out):
    read=lambda p:json.loads(p.read_text())
    work=Path(__file__).resolve().parent
    lower,walk=native_functions(surface_lib,work/"trace_columns.dylib")
    pre="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    env=read(dense/"launch.json")["environment"]; identities={}
    def pinned(key,identity=None):
        p=Path(env[pre+key]);h=digest(p)
        if h!=env[pre+(identity or key+"_SHA256")]:raise ValueError("Changed source "+key)
        identities[key]=h;return p
    reference=pinned("TETGEN_NODE"); topology=pinned("TETGEN_ELEMENTS")
    mapping=pinned("COMBINED_NODE_MAP_CSV","COMBINED_NODE_MAP_SHA256")
    node_rows=np.loadtxt(reference,skiprows=1,usecols=(0,1,2,3))
    if not np.array_equal(node_rows[:,0],np.arange(1,1+len(node_rows))):raise ValueError("Unknown node order")
    source=np.ascontiguousarray(node_rows[:,1:].astype("f4").astype(float))
    cell_rows=np.loadtxt(topology,skiprows=1,dtype="i4")
    if cell_rows.shape[1]!=6 or not np.array_equal(cell_rows[:,0],np.arange(1,len(cell_rows)+1)):
        raise ValueError("Unknown cell layout")
    cells=np.ascontiguousarray(cell_rows[:,1:5]-1);mat=np.ascontiguousarray(cell_rows[:,5])
    base_cells=len(cells)
    cells,mat=installed_topology(cells,mat,env)
    cells=np.ascontiguousarray(cells,dtype='i4');mat=np.ascontiguousarray(mat,dtype='i4')
    cavity=Path(env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ'])
    identities['CAVITY_REPLACEMENTS']=digest(cavity)
    identities['INSTALLED_TETRAHEDRA']=hashlib.sha256(cells.astype('<i4').tobytes()).hexdigest()
    current_path=Path(env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE'])
    identities['INSTALLED_CURRENT_NODES']=digest(current_path)
    if identities['INSTALLED_CURRENT_NODES']!=read(dense/'launch.json')['pins'][str(current_path)]:raise ValueError('Unpinned deformed state')
    current_rows=np.loadtxt(current_path,skiprows=1,usecols=(0,1,2,3))
    if not np.array_equal(current_rows[:,0],node_rows[:,0]):raise ValueError('Current material node order drift')
    current=np.ascontiguousarray(current_rows[:,1:].astype('f4').astype(float))
    receipt=Path(env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_RECORDED_ALLOCATION_SKIN_RECEIPT_JSON'])
    flow=receipt.parents[2]/'fine-mass-routed/current-density-closed-flow.npz'
    identities['MASS_FLOW']=digest(flow)
    if identities['MASS_FLOW']!=read(flow.parent/'mass-transfer.json')['flow_sha256']:raise ValueError('Unpinned mass')
    with np.load(flow) as ledger,np.load(cavity) as patch:
        mass=np.r_[ledger['element_mass_kg'][patch['retained_input_cell_rows']],patch['replacement_mass_kg']]
    nodal=np.bincount(cells.ravel(),np.repeat(mass*.25,4),minlength=len(source))
    if len(mass)!=len(cells) or min(mass.min(),nodal.min())<=0 or abs(mass.sum()-nodal.sum())>1e-10:raise ValueError('Mass mapping failed')
    mass_by_role={k:float(mass[mat==m].sum()) for k,m in (('skin_kg',15),('fat_kg',16))}
    mass_by_role.update(total_kg=float(mass.sum()),nodal_kg=float(nodal.sum()))
    old={b["body_index"]:b["current"] for b in read(dense/"registered-initial-bodies.json")["all_bodies"]}
    reg=read(opening/"terminal.receipt.json")["initial_anatomical_registration"]
    new={b["body_index"]:b for b in reg["body_poses"]}
    if old.keys()!=new.keys():
        launch=read(opening/"launch.json")
        frame_file=Path(launch["environment"]["NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP"])
        frames=read(frame_file)["all_bodies"]
        actual={v["body_index"]:dict(body_index=v["body_index"],quaternion_xyzw=v["current"]["orientation_world_xyzw"],
                                   position_m=v["current"]["position_world_m"]) for v in frames}
        for b,r in new.items():
            if np.max(abs(np.array(r["position_m"])-actual[b]["position_m"]))>4e-7 or abs(np.dot(r["quaternion_xyzw"],actual[b]["quaternion_xyzw"]))<1-2e-7:
                raise ValueError("Native opening pose differs between receipts")
        identities["NATIVE_ALL_BODY_OPENING"]=digest(frame_file);new=actual
    if sorted(old)!=list(range(len(old))) or old.keys()!=new.keys():raise ValueError("Different skeletons")
    def transform(table,rot,pos):
        return Rotation.from_quat([table[i][rot] for i in range(len(table))]).as_matrix(),np.array([table[i][pos] for i in range(len(table))])
    a,x=transform(old,"orientation_world_xyzw","position_world_m");b,y=transform(new,"quaternion_xyzw","position_m")
    indices={k:{} for k in "OID"};donor=np.full(len(source),-1,"i4");layers=np.zeros(len(source),"i4")
    # Source has two material bookkeeping rows at the same shared interface.
    with mapping.open() as f:
        for row in csv.DictReader(f):
            node=int(row["joint_node_id"])-1
            owner=int(row["donor_body"])
            if not (0<=node<len(source) and owner in old):raise ValueError(("Unbound material node",node,owner))
            if donor[node]!=-1 and donor[node]!=owner:raise ValueError("Conflicting material owners")
            donor[node]=owner
            k=row["registered_surface_layer"]
            if k not in indices:continue
            local=int(row["registered_surface_vertex_row"]); prior=indices[k].setdefault(local,node)
            if prior!=node:raise ValueError("Conflicting boundary ancestry")
            layers[node]="OID".index(k)+1
    if donor.min()<0:raise ValueError("Missing volume node binding")
    posed=np.einsum("nij,nj->ni",(b@a.transpose(0,2,1))[donor],source-x[donor])+y[donor]
    relative=b@a.transpose(0,2,1)
    translation=y-x@relative[0].T
    angular=float(Rotation.from_matrix(relative@relative[0].T).magnitude().max())
    origin_error=float(np.linalg.norm(translation-translation[0],axis=1).max())
    epoch=source@relative[0].T+translation[0]
    global_error=float(np.linalg.norm(epoch-posed,axis=1).max())
    rigid_equivalent=(angular<1e-6 and origin_error<1e-6 and global_error<1e-6)
    if rigid_equivalent: posed=np.ascontiguousarray(epoch)
    current_posed=(current@relative[0].T+translation[0] if rigid_equivalent else
                   np.einsum("nij,nj->ni",relative[donor],current-x[donor])+y[donor])
    registration=dict(global_rotation_residual_rad=angular,global_origin_residual_m=origin_error,
                      global_node_residual_m=global_error,single_rigid_transform=rigid_equivalent)
    epochs={}
    for k in "OID":
        v=indices[k]
        if sorted(v)!=list(range(len(v))):raise ValueError("Unordered boundary")
        nodes=np.array([v[i] for i in range(len(v))],dtype="i4")
        f=np.loadtxt(pinned("SURFACE_"+k+"_FACES_CSV","SURFACE_"+k+"_FACES_SHA256"),delimiter=",",skiprows=1,usecols=(1,2,3),dtype="i4")
        epochs[k]=np.ascontiguousarray(nodes[f])
    # Float32 reference coordinates are the same ones carried by the solver.
    deformations=[];emitted=[];loaded=[];quantized_loaded=[]
    for start in range(0,len(cells),50000):
        seg=cells[start:start+50000];ref=source[seg];cur=posed[seg]
        v0=np.linalg.det(ref[:,1:]-ref[:,:1]);v1=np.linalg.det(cur[:,1:]-cur[:,:1])
        if v0.min()<=0:raise ValueError(("Invalid reference tetrahedron",start+int(v0.argmin()),v0.min()))
        deformations.append(v1/v0)
        quantized=cur.astype("f4").astype(float);emitted.append(np.linalg.det(quantized[:,1:]-quantized[:,:1])/v0)
        loaded_block=current_posed[seg]
        for target,block in ((loaded,loaded_block),(quantized_loaded,loaded_block.astype('f4').astype(float))):
            target.append(np.linalg.det(block[:,1:]-block[:,:1])/v0)
    j_loaded=np.concatenate(loaded);j_loaded_f32=np.concatenate(quantized_loaded)
    if not all(np.isfinite(x).all() for x in (j_loaded,j_loaded_f32)):raise ValueError('Nonfinite deformed material')
    jac=np.concatenate(deformations);qjac=np.concatenate(emitted)
    proof=read(case/"terminal-bed-certificate.json");path=case/"registered-bed-columns.csv"
    if digest(path)!=proof["columns"]:raise ValueError("Stale material columns")
    data=np.genfromtxt(path,delimiter=",",names=True)
    data=data[data["area_m2"].astype("f4")>0]
    n=len(data);xy=np.ascontiguousarray(np.stack([data["x_m"],data["y_m"]],axis=1))
    if n!=proof["checked_columns"]:raise ValueError("Incomplete physical footprint")
    first=np.empty(n,"i4");z=np.empty(n);beta=np.empty((n,3))
    if lower(posed,len(posed),epochs["O"],len(epochs["O"]),xy,n,first,z,beta):raise ValueError("Outer boundary lookup failed")
    valid=first>=0
    hit=(beta[valid,:,None]*posed[epochs["O"][first[valid]]]).sum(axis=1)
    if abs(hit-np.c_[xy[valid],z[valid]]).max()>1e-8:raise ValueError("Bad outer face ancestry")
    info=np.empty((n,5),"i4"); h=np.empty((n,4))
    code=walk(source,posed,len(posed),cells,mat,len(cells),layers,epochs["O"],len(epochs["O"]),first,xy,z,n,info,h)
    if code:raise ValueError(("Material traversal error",code))
    reached=info[:,0]==0
    if np.any(h[reached,2]<=h[reached,1]) or np.any(h[reached,1]<=h[reached,0]):raise ValueError("Nonpositive layer length")
    # Ensure the I and D faces reached by tetrahedral adjacency exist in the emitted layer mesh.
    def keys(f):
        a=np.sort(f,axis=1).astype("u8");return (a[:,0]<<40)|(a[:,1]<<20)|a[:,2]
    boundary={}
    for k,slot in (("I",2),("D",3)):
        ids=info[reached,slot];f=cells[(ids//4)[:,None],((ids%4)[:,None]+np.array([1,2,3]))%4]
        src=keys(epochs[k]);order=np.argsort(src);hits=keys(f);where=np.searchsorted(src[order],hits)
        if np.any(where==len(src)) or not np.array_equal(src[order[where]],hits): raise ValueError("Traversal left declared "+k+" mesh")
        if np.unique(src).size!=len(src):raise ValueError("Duplicate boundary face")
        boundary[k]=np.full(n,-1,"i4");boundary[k][reached]=order[where]
    incidence=np.zeros((n,3))
    for j,(name,idx) in enumerate((("O",first),("I",boundary["I"]),("D",boundary["D"]))):
        tri=posed[epochs[name][idx[reached]]]
        normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        incidence[reached,j]=abs(normal[:,2])/np.linalg.norm(normal,axis=1)
    owners=np.full(n,-1,"i4"); owners[valid]=donor[epochs["O"][first[valid],np.argmax(beta[valid],axis=1)]]
    np.savez_compressed(out/"paths.npz",vertices=data["vertex"].astype("i4"),rows=data["row"].astype("i4"),
        area=data["area_m2"],source_xyz=np.c_[xy,data["z_m"]],old_skin=data["skin_m"],old_fat=data["fat_m"],
        old_source=data["thickness_source_vertex"].astype("i4"),outer_face=first,outer_owner=owners,bary=beta,
        result=info,heights=h,vertical_normal_cosine=incidence,interface_face=boundary["I"],deep_face=boundary["D"])
    labels=["through_skin_and_fat","no_outer_face","unowned_outer","invalid_posed_cell","tangent_or_ambiguous",
            "exited_before_deep_face","material_order","cycle"]
    reasons={reason:int(np.count_nonzero(info[:,0]==i)) for i,reason in enumerate(labels)}
    sample=np.flatnonzero(data["vertex"]==17764).item()
    terminal=dict(source_vertex=17764,status=labels[info[sample,0]],tet_count=int(info[sample,1]),heights_m=h[sample,:3].tolist(),
                  minimum_determinant_ratio=float(h[sample,3]),original_skin_m=float(data["skin_m"][sample]),original_fat_m=float(data["fat_m"][sample]))
    report=dict(registration=registration,base_cells=base_cells,mass=mass_by_role,
        loaded_state=dict(inversions=int(sum(j_loaded<=0)),float32_inversions=int(sum(j_loaded_f32<=0)),
                          min_J=float(j_loaded.min()),float32_min_J=float(j_loaded_f32.min())),
        float32_inversions=int(sum(qjac<=0)),float32_min_J=float(qjac.min()),cells=len(cells),nodes=len(posed),source_identities=identities,datum_pose=digest(opening/"terminal.receipt.json"),
        enrolled_columns=n,traversal_counts=reasons,projected_area_m2=float(data["area_m2"].sum()),traversed_area_m2=float(data["area_m2"][reached].sum()),
        worst_jacobian_ratio=float(jac.min()),inverted_after_rigid_registration=int(np.count_nonzero(jac<=0)),
        posed_cell_ratios={str(m):np.quantile(jac[mat==m],[0,.01,.5,.99,1]).tolist() for m in (15,16)},shoulder=terminal,
        reference_columns=digest(path),paths=digest(out/"paths.npz"),scope="Authoring audit of registered material connectivity in the opening pose. No physical state advances.")
    (out/"connected-columns.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    prepare(*(Path(v).resolve() for v in sys.argv[1:]))
