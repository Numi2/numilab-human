"""Complete a cortical compliance atlas for every material parent of a registered moving facet."""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation

from material_context import sha
from pelvic_elastic_response import stiffness
from qualify_material_motion import frame

INVALID = np.iinfo(np.uint32).max
OUTPUTS = {"cortical-free-index.bin", "cortical-flexibility.bin", "manifest.json"}

def read_surface(assets, specs):
    triangles = np.fromfile(assets / "closed-material-triangles.bin", "<u4").reshape(-1,3,4)
    shapes = np.fromfile(assets / "closed-material-shapes.bin", "<f4").reshape(-1,3,4)
    if not np.isfinite(shapes).all() or np.max(abs(shapes.sum(axis=2)-1)) > 1.e-5:
        raise ValueError("Invalid material surface shape functions")
    for side, region in enumerate(specs["contact_flexibility"]):
        faces = specs["closed_reverse"]["sides"][side]
        span = slice(faces["first_face"], faces["first_face"]+faces["faces"])
        parents = triangles[span,2]
        # A linear combination of the three face corners cannot use a
        # material parent whose coefficient is zero at all three corners.
        active = (shapes[span] != 0).any(axis=1)
        ids = np.unique(parents[active]).astype(int) - region["first_node"]
        if ids.min() < 0 or ids.max() >= region["nodes"]:
            raise ValueError("Surface interpolation crosses a material owner")
        yield ids

def extend_side(part, assets, specs, side, required, lookup, old, packed, rotation):
    region = specs["contact_flexibility"][side]
    sid = region["muscle"]
    x0 = np.load(part/"projected"/f"{sid}.npz")
    mat = np.load(part/"response"/f"{sid}.npz")
    xyz = x0["vertices"].astype(float)
    x = rotation.apply(xyz-xyz.mean(axis=0))
    positive = x0["cell_volumes"] > 0
    cells = x0["cells"][positive]
    A,_ = stiffness(x,cells,x0["cell_volumes"][positive],specs["specimen"]["mu"],specs["specimen"]["lam"])
    free = np.setdiff1d(np.unique(cells),mat["fixed"])
    dofs = (3*free[:,None]+np.arange(3)).ravel()
    K = A[dofs][:,dofs].tocsc()
    unit = 1 / np.sqrt(K.diagonal())
    D = sparse.diags(unit)
    H = (D@K@D).tocsc()
    patch = sparse.lil_matrix(H.shape)
    quotient = []
    for group in specs["regional_factor"][side]["floating_material_components"]:
        nodes = np.asarray(group["nodes"])
        ix = np.searchsorted(dofs,(3*nodes[:,None]+np.arange(3)).ravel())
        rel = x[nodes]-x[nodes].mean(axis=0)
        modes = np.zeros((len(nodes),3,6))
        modes[:,:,:3] = np.eye(3)
        for k in range(3):
            modes[:,:,3+k] = np.cross(np.eye(3)[k],rel)
        Q = np.linalg.qr(modes.reshape(-1,6)/unit[ix,None],mode="reduced")[0]
        if np.linalg.norm(H[ix][:,ix]@Q,np.inf) > 1e-8:
            raise ValueError("Unsupported material rigid modes")
        patch[np.ix_(ix,ix)] = Q@Q.T
        quotient.append((ix,Q))
    solver = splu((H + patch.tocsc()).tocsc())
    def solve(forces):
        rhs = unit[:,None]*forces
        for ix,Q in quotient:
            rhs[ix] -= Q@(Q.T@rhs[ix])
        u = solver.solve(rhs)
        for ix,Q in quotient:
            u[ix] -= Q@(Q.T@u[ix])
        return unit[:,None]*u, rhs/unit[:,None]

    missing = required[lookup[required] == INVALID]
    count = region["input_nodes"]
    if not np.array_equal(np.sort(lookup[lookup != INVALID]),np.arange(count)):
        raise ValueError("Source input map is not dense in its registered block")
    packed[:,:count] = old
    idx = np.full(3*len(x),-1)
    idx[dofs] = np.arange(len(dofs))
    largest = 0.
    for start in range(0,len(missing),12):
        parents = missing[start:start+12]
        dest = count+start
        indices = idx[(3*parents[:,None]+np.arange(3)).ravel()]
        rhs = np.zeros((len(dofs),len(indices)))
        live = np.flatnonzero(indices>=0)
        rhs[indices[live], live] = 1
        disp, balanced = solve(rhs)
        largest = max(largest,float(np.max(abs(K@disp-balanced))))
        full = np.zeros((len(x)*3,len(indices)))
        full[dofs] = disp
        packed[:,dest:dest+len(parents),:,:3] = full.reshape(len(x),3,len(parents),3).transpose(0,2,3,1)
        packed[:,dest:dest+len(parents),:,3] = 0
        lookup[parents] = dest+np.arange(len(parents))
    if largest>1e-6 or not np.array_equal(packed[:,:count],old):
        raise ValueError(f"Material parity or force response failed: {largest}")
    # Independent action and stiffness-energy test on a mixed new/old load.
    rng = np.random.default_rng(sid)
    weights = rng.normal(size=(len(required),3))
    rhs = np.zeros((len(x),3))
    rhs[required] = weights
    exact,_ = solve(rhs.ravel()[dofs,None])
    u = np.zeros((len(x),3))
    for nodes, loads in zip(np.array_split(required,12),np.array_split(weights,12)):
        u += np.einsum("nskd,sk->nd",packed[:,lookup[nodes],:,:3].astype(float),loads)
    error = u.ravel()[dofs] - exact[:,0]
    rel = np.sqrt(max(0.,error@(K@error))/(exact[:,0]@(K@exact[:,0])))
    if not np.isfinite(rel) or rel>1e-3 or (lookup[required] == INVALID).any():
        raise ValueError(f"Packed compliance failed mixed-load verification: {rel}")
    return dict(muscle=sid,mapped_parent_nodes=len(required),new_columns=len(missing),
                projected_source_sha256=sha(part/"projected"/f"{sid}.npz"),
                attachment_source_sha256=sha(part/"response"/f"{sid}.npz"),
                previous_columns=count,unchanged_previous_values=True,
                worst_new_double_force_error_N=largest,mixed_float32_relative_energy_error=float(rel),
                free_floating_components=len(quotient))

def build(part,assets,pose,out):
    if out.exists():
        raise ValueError("Output must be a new asset directory")
    specs=json.loads((assets/"manifest.json").read_text())
    for k in sorted((OUTPUTS-{"manifest.json"}) | {"closed-material-triangles.bin","closed-material-shapes.bin"}):
        if sha(assets/k) != specs["files"][k]:
            raise ValueError("Source identity mismatch: "+k)
    roots = json.loads(pose.read_text())["initial_anatomical_registration"]["body_poses"]
    rotation=Rotation.from_quat(frame(next(x for x in roots if x["body_index"]==128))[3:]).inv()
    required=list(read_surface(assets,specs))
    original=np.memmap(assets/"cortical-flexibility.bin",dtype="<f4",mode="r").reshape(-1,4)
    lookup=np.fromfile(assets/"cortical-free-index.bin","<u4")
    counts=[]; report=[]
    for region,need in zip(specs["contact_flexibility"],required):
        local=lookup[region["first_node"]:][:region["nodes"]]
        counts.append(region["input_nodes"]+np.count_nonzero(local[need]==INVALID))
    with TemporaryDirectory(dir=out.parent) as tmp:
        root=Path(tmp)/"assets"
        root.mkdir()
        for f in assets.iterdir():
            if f.is_file() and f.name not in OUTPUTS:os.link(f,root/f.name)
        cache=root/"cortical-flexibility.bin"
        with cache.open("wb") as stream:
            stream.truncate(sum(48*r["nodes"]*int(n) for n,r in zip(counts,specs["contact_flexibility"])))
        base=0
        for side,(new,region,need) in enumerate(zip(counts,specs["contact_flexibility"],required)):
            n=region["nodes"]; count=region["input_nodes"]
            old=original[region["first_vector"]:][:3*n*count].reshape(n,count,3,4)
            packed=np.memmap(cache,dtype="<f4",mode="r+",offset=16*base,shape=(n,int(new),3,4))
            local=lookup[region["first_node"]:][:n]
            report.append(extend_side(part,assets,specs,side,need,local,old,packed,rotation))
            packed.flush()
            region.update(input_nodes=int(new),first_vector=base)
            base+=3*n*int(new)
        lookup.tofile(root/"cortical-free-index.bin")
        for name in OUTPUTS-{"manifest.json"}:specs["files"][name]=sha(root/name)
        specs["surface_force_coverage"]={"parent_manifest":sha(assets/"manifest.json"),
                                         "registration_receipt_sha256":sha(pose),"sides":report}
        (root/"manifest.json").write_text(json.dumps(specs,indent=2)+"\n")
        root.rename(out)
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("partition","assets","pose","output"):
        parser.add_argument(key,type=Path)
    args=parser.parse_args()
    build(args.partition,args.assets,args.pose,args.output)
