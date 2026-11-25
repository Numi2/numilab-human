"""Register newly refined deep-fat nodes with their source faces, skeletal frames and parent tetrahedra."""

import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from refine_deep_fat_boundary import PREFIX, digest, face_key


def text(numbers):
    return [format(float(n), ".17g") for n in numbers]


def closest_tet(point, vertices):
    """Return the closest point of the closed simplex in barycentric coordinates."""
    best = np.inf
    coords = None
    for size in range(4, 0, -1):
        for ids in combinations(range(4), size):
            x = vertices[list(ids)]
            u = np.linalg.lstsq((x[1:] - x[0]).T, point-x[0], rcond=None)[0]
            coeff = np.r_[1 - u.sum(), u]
            if coeff.min() < -1e-10:
                continue
            coeff = np.maximum(coeff, 0)
            coeff /= coeff.sum()
            d2 = np.sum((point-coeff @ x)**2)
            if d2 < best:
                best = d2
                coords = np.zeros(4)
                coords[list(ids)] = coeff
            if size == 4:
                return best, coords
    return best, coords


def run(launch, refined, label_path, out):
    config = json.loads(launch.read_bytes())
    env = config["environment"]
    def path(key):
        f = Path(env[key])
        if digest(f) != config["pins"].get(str(f), env.get(key+"_SHA256") or env.get(key.removesuffix("_CSV")+"_SHA256")):
            raise ValueError("Changed source " + key)
        return f
    def old(key):
        return path(PREFIX + key + "_CSV")
    receipt = json.loads((out/"deep-boundary.json").read_bytes())
    data = out/"deep-registration.npz"
    for name, sha in receipt["outputs"].items():
        if digest(out/name) != sha:
            raise ValueError("Changed boundary transfer " + name)
    if digest(launch) != receipt["source_launch_sha256"] or digest(refined/"mesh.ele") != receipt["element_sha256"]:
        raise ValueError("Mismatched volume or registered parent")
    m = np.load(data)
    n = receipt["old_nodes"]
    nodes = m["nodes"]
    P = csr_matrix((m["weights"], m["indices"], m["indptr"]), shape=(len(nodes)-n,n))
    xy = np.loadtxt(refined/"mesh.node",skiprows=1,usecols=(1,2,3))
    t = np.loadtxt(refined/"mesh.ele",skiprows=1,usecols=(1,2,3,4),dtype=int)-1
    labels = np.loadtxt(label_path,skiprows=1,delimiter=",",dtype=int)
    if not np.array_equal(labels[:,0],np.arange(len(t))):
        raise ValueError("Incomplete physical tissue labels")
    with np.load(refined/"source-overlay.npz") as overlay:
        child = overlay["geometric_child"]
        ancestor = overlay["geometric_parent"]
    by_parent = defaultdict(list)
    by_node = defaultdict(list)
    lookup = {}
    for parent, k in zip(ancestor, child, strict=True):
        by_parent[parent].append(k)
        for j, q in enumerate(t[k]):
            by_node[q].append((k,j))
        for f in combinations(t[k],3):
            lookup[parent,face_key(f)] = k
    new_faces = np.loadtxt(out/"surface_d_faces.csv",skiprows=1,delimiter=",",usecols=(1,2,3,4,5),dtype=int)
    base_faces = np.loadtxt(old("SURFACE_D_FACES"),skiprows=1,delimiter=",",usecols=(1,2,3,4,5),dtype=int)
    new_by_old = defaultdict(list)
    for idx in np.flatnonzero(np.isin(m["parents"], np.flatnonzero(np.bincount(m["parents"])>1))):
        original = m["parents"][idx]
        new_by_old[face_key(nodes[base_faces[original,:3]])].append(nodes[new_faces[idx,:3]])
    points = np.loadtxt(path("NUMI_H706_FRACTIONAL_SOURCE330_NODES"),skiprows=1,delimiter=",",usecols=(2,3,4,5,6,7))
    pt = np.loadtxt(path("NUMI_H706_FRACTIONAL_SOURCE330_TETS"),skiprows=1,delimiter=",",usecols=(3,4,5,6),dtype=int)
    pl = np.loadtxt(old("PARENT_LABELS"),skiprows=1,delimiter=",",dtype=int)
    pv = points[pt,3:]
    boxlo = pv.min(axis=1)
    boxhi = pv.max(axis=1)
    with old("DONOR_FRAMES").open() as f:
        frame = {int(v["binding"]): v for v in csv.DictReader(f)}
    with (out/"role_d_full86.csv").open() as f:
        next(f)
        weights = [np.fromstring(line,sep=",")[1:] for i,line in enumerate(f) if i>=n]
    source = json.loads(path(PREFIX+"ROLE_D_SOURCE_ASSET").read_bytes())
    sid = np.array([s["source_vertex_row"] for s in source["vertices"]])
    sv = np.array([s["current_world_m"] for s in source["vertices"]])
    sf = np.array([s["surface_vertex_rows"] for s in source["faces"]])
    old_ids = np.loadtxt(old("SURFACE_D_NODES"),skiprows=1,delimiter=",",usecols=30,dtype=int)
    if np.any(old_ids[P.indices] < 0):
        raise ValueError("Nested source-face reconstruction required")
    original = np.flatnonzero(np.bincount(m["parents"])>1)
    candidates = sorted(set(base_faces[original,3]))
    surface_add = []
    reg_add = []
    map_add = []
    mx_source_error = 0.
    mx_hist_dist = 0.
    mx_ref_dist = 0.
    norm = dict(zip(m["patch"],m["normal"],strict=True))
    for i,node in enumerate(nodes[n:]):
        rr = n+i
        indices, coeff = P.indices[P.indptr[i]:P.indptr[i+1]], P.data[P.indptr[i]:P.indptr[i+1]]
        src_ids = old_ids[indices]
        matches = [k for k in candidates if np.isin(src_ids,sid[sf[k]]).all()]
        if not matches:
            raise ValueError(f"Node {node} has no common original source triangle")
        face = matches[0]
        b = np.zeros(3)
        for key,val in zip(src_ids,coeff,strict=True):
            j, = np.flatnonzero(sid[sf[face]]==key)
            b[j] += val
        q = xy[node]
        residual = q-b@sv[sf[face]]
        mx_source_error = max(mx_source_error,np.linalg.norm(residual))
        # Compare against inherited source offsets, plus the emitted Float32 placement.
        inherited = coeff@(xy[nodes[indices]] - sv[sf[face]][[int(np.flatnonzero(sid[sf[face]]==z)[0]) for z in src_ids]])
        if np.linalg.norm(residual-inherited) > receipt["max_position_error_m"]+1e-12:
            raise ValueError("New source-face error exceeds material interpolation error")
        k,slot = min(by_node[node])
        _, mat, donor, region, _ = labels[k]
        if mat!=16:
            raise ValueError("Deep boundary touches non-fat material")
        # Source and operational reference are different registered meshes.
        # Search the source volume in the tissue domain. At identical distances
        # prefer this face's declared source parent, then its operational region.
        valid = np.flatnonzero(pl[:,1]==mat)
        if not len(valid):
            raise ValueError("No historical material group for new boundary node")
        dist_box = np.maximum(np.maximum(boxlo[valid]-q, q-boxhi[valid]), 0)
        ds = np.sum(dist_box**2,axis=1)
        best = np.inf
        source_tet = source["faces"][face]["parent_tet_row"]
        for j in np.lexsort((valid, (pl[valid,2]!=donor)|(pl[valid,3]!=region), valid!=source_tet, ds)):
            if ds[j] > best+1e-22:
                break
            d2, bary = closest_tet(q, pv[valid[j]])
            if d2 < best:
                best, tc, bc = d2, valid[j], bary
        source_residual = q-bc @ points[pt[tc],:3]
        hist_dist = np.linalg.norm(source_residual)
        mx_hist_dist = max(mx_hist_dist,hist_dist)
        mx_ref_dist = max(mx_ref_dist,best**.5)
        binding = int(np.argmax(weights[i]))
        ref = frame[binding]
        rot = np.array([float(ref[f"current_R{a}{j}"]) for a in range(3) for j in range(3)]).reshape(3,3)
        orig = np.array([float(ref["current_"+axis]) for axis in "xyz"])
        transform = np.linalg.inv(rot)
        local = transform @ (q-orig)
        if np.linalg.norm(rot@local+orig-q)>1e-12:
            raise ValueError("Invalid donor rigid frame")
        source_role = env[PREFIX+"ROLE_D_SOURCE_DOMAIN"]
        method = "refined_deep_face_interpolated_full86_donor"
        embedding = ["-1",str(face),*text(b),*text(residual)]
        pr = ["source_parent_tet_reference_registration",str(tc),*text(bc),*text(source_residual),*text([hist_dist]),"0"]
        surface_add.append([rr,ref["donor"],*text(q),*text(norm[rr]),*text(local),*text(transform.ravel()),source_role,*embedding,method,"-1"])
        reg_add.append(["D",rr,*pr,method,source_role,*embedding])
        map_add.append([int(node)+1,"fat",None,-1,"","","D",rr,"D",*text(q),ref["donor"],binding,*text(local),
                        source_role,"face",*embedding,method,*pr,1<<16,donor,region,-1,k,
                        *[int(j==slot) for j in range(4)],0,0,0,0,pl[tc,2],pl[tc,3]])
    with old("SURFACE_D_NODES").open() as src,(out/"surface_d_nodes.csv").open("w")as dst:
        dst.write(src.readline())
        for i,line in enumerate(src):
            if i in norm:
                parts=line.rstrip().split(",")
                parts[5:8] = text(norm[i])
                line = ",".join(parts)+"\n"
            dst.write(line)
        csv.writer(dst,lineterminator="\n").writerows(surface_add)
    with old("NODE_REGISTRATION").open() as src,(out/"node_registration.csv").open("w")as dst:
        for line in src:
            dst.write(line)
        csv.writer(dst,lineterminator="\n").writerows(reg_add)
    nchanged = 0
    partition_max = 0
    with old("COMBINED_NODE_MAP").open() as src,(out/"combined_node_map.csv").open("w")as dst:
        dst.write(src.readline())
        for line in src:
            r=line.rstrip().split(",")
            if r[1]=="fat":
                partition_max=max(partition_max,int(r[2]))
            parent=int(r[43])
            if parent in by_parent:
                node=int(r[0])-1
                hit=[(k,slot) for k in by_parent[parent] for slot in np.flatnonzero(t[k]==node)]
                if not hit or any(float(v)for v in r[48:52]):
                    raise ValueError("Original node has no exact operational child on its source tet")
                k,slot=min(hit)
                r[43:48]=[str(k),*[str(int(j==slot))for j in range(4)]]
                line=",".join(r)+"\n"
                nchanged+=1
            dst.write(line)
        for i,r in enumerate(map_add):
            r[2]=partition_max+1+i
        csv.writer(dst,lineterminator="\n").writerows(map_add)
    extra = []
    largest = -1
    seen_faces = set()
    with old("FACET_ANCESTRY").open() as src,(out/"facet_ancestry.csv").open("w")as dst:
        reader=csv.reader(src)
        writer=csv.writer(dst,lineterminator="\n")
        writer.writerow(next(reader))
        for r in reader:
            if r[0]=="fat":
                largest=max(largest,int(r[1]))
            parent=int(r[18])
            if parent not in by_parent:
                writer.writerow(r)
                continue
            corners=np.array(r[19:22],int)-1
            key=face_key(corners)
            children = new_by_old.get(key, [corners])
            if len(children)>1:
                if r[3]!="inner_D" or key in seen_faces:
                    raise ValueError("Unexpected refined boundary owner")
                seen_faces.add(key)
            for i,q in enumerate(children):
                k=lookup[parent,face_key(q)]
                v=r.copy()
                v[17:22]=[str(k+1),str(k),*[str(a+1) for a in q]]
                if i:
                    extra.append(v)
                else:
                    writer.writerow(v)
        for i,v in enumerate(extra):
            v[1]=str(largest+1+i)
            writer.writerow(v)
    if seen_faces!=set(new_by_old):
        raise ValueError("Missing refined face ancestry")
    outputs = ["surface_d_nodes.csv","node_registration.csv","combined_node_map.csv","facet_ancestry.csv"]
    result=dict(registered_nodes=len(surface_add),new_facet_ancestry_rows=len(extra),
                updated_original_node_addresses=nchanged,max_source_face_offset_m=float(mx_source_error),
                max_historical_parent_offset_m=float(mx_hist_dist),max_reference_projection_m=float(mx_ref_dist),
                source_face_labels_sha256=digest(label_path),outputs={name:digest(out/name) for name in outputs})
    (out/"physical-registration.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    run(*(Path(p) for p in sys.argv[1:]))
