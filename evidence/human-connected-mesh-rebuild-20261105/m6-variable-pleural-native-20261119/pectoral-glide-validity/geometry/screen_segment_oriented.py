"""Remove an exact edge cut using local segment/triangle separating planes."""
from pathlib import Path

prior_code = Path(__file__).with_name("screen_fixed_edge_interior_lp.py").read_text()
exec(compile(prior_code[:prior_code.index("def fixed_edge_separator")], __file__, "exec"))

pose_edge = posed_skin[np.searchsorted(skin_rows, sorted(edge_vertices))].astype(float)
edge_dir = pose_edge[1] - pose_edge[0]
average_normal = area(posed_skin[emitted_faces[edge_faces]].astype(float)).sum(axis=0)
average_normal /= np.linalg.norm(average_normal)
orientation = vstack([protect_face_orientation(points, None),
                      protect_face_orientation(posed, world_jac)])
dim = len(mobile)*3
identity = eye(dim, format="csr")
equal_volume = np.r_[grads, np.zeros(dim)].reshape(1,-1)

def choices(triangle):
    corners = posed[faces[triangle]].astype(float)
    edges = np.roll(corners,-1,axis=0)-corners
    axes = np.vstack([np.cross(edges[0],edges[1]), np.cross(edge_dir,edges), average_normal])
    norms = np.linalg.norm(axes,axis=1)
    axes = axes[norms>1.e-12]/norms[norms>1.e-12,None]
    axes = np.concatenate([axes,-axes])
    # Choose the shortest separating correction among outward-compatible axes.
    edge_floor = ((pose_edge-pose_edge[0])@axes.T).min(axis=0)
    gaps = 1000.*(((corners-pose_edge[0])@axes.T)-edge_floor)
    agreement = axes@average_normal
    cost = np.maximum(gaps+.03,0.).max(axis=0) + .15*(1.-agreement)
    return [(axes[k],gaps[:,k]) for k in np.argsort(cost) if agreement[k]>=-.15]

def constrain(chosen):
    rhs = []
    vals = []
    cols = []
    for triangle, (direction, gaps) in chosen.items():
        for corner, source_row in enumerate(faces[triangle]):
            target_gap = -.03-gaps[corner]
            if local[source_row] < 0:
                if target_gap < 0.:
                    return None
                continue
            cols.append(3*local[source_row]+np.arange(3))
            vals.append(direction@world_jac[source_row])
            rhs.append(target_gap)
    op = coo_matrix((np.ravel(vals),(np.repeat(np.arange(len(rhs)),3),np.ravel(cols))),
                    shape=(len(rhs),dim)).tocsr()
    return op, rhs

results = []
for bound, orientation_limit in ((1.4,.125),(1.4,.115),(1.4,.1),(1.7,.09)):
    chosen = {}
    remaining = base_cuts
    for attempt in range(8):
        for triangle in remaining:
            if triangle in chosen:
                break
            options = choices(triangle)
            if options:
                chosen[triangle]=options[0]
        linear = constrain(chosen)
        if linear is None:
            results.append(dict(bound=bound,attempt=attempt,result="protected attachment"))
            break
        op,rhs=linear
        shape = vstack([-orientation,op])
        inequalities = vstack([hstack([shape,coo_matrix((shape.shape[0],dim))]),
                               hstack([identity,-identity]),hstack([-identity,-identity])]).tocsr()
        limits=np.r_[np.full(orientation.shape[0],orientation_limit),rhs,np.zeros(2*dim)]
        solution=linprog(np.r_[np.zeros(dim),np.ones(dim)], A_ub=inequalities,b_ub=limits,
                         A_eq=equal_volume,b_eq=[0.],
                         bounds=[(-bound,bound)]*dim+[(0.,bound)]*dim,method="highs")
        if not solution.success:
            results.append(dict(bound=bound,attempt=attempt,result=solution.message))
            break
        offset=np.zeros_like(points)
        offset[mobile]=solution.x[:dim].reshape(-1,3)*.001
        local_trial=(points+offset).astype("<f4")
        world_trial=(posed+np.einsum("vij,vj->vi",world_jac,offset)).astype("<f4")
        rest_normal,rest_volume=quality(points,local_trial.astype(float),faces)
        posed_normal,posed_volume=quality(posed.astype(float),world_trial.astype(float),faces)
        record=dict(bound=bound,attempt=attempt,orientation_limit=orientation_limit,rest_normal=rest_normal,rest_volume=rest_volume,
                    posed_normal=posed_normal,posed_volume=posed_volume,
                    max_move_m=float(np.linalg.norm(offset,axis=1).max()),
                    constrained_triangles=sorted(chosen))
        results.append(record)
        if min(rest_normal,posed_normal)<.85 or max(rest_volume,posed_volume)>.005:
            record["result"]="finite face quality"
            break
        after=exact_skin_pair(exact_index,float_index,world_trial,faces)
        remaining=obstruction(world_trial,after)
        metrics_after=curves(world_trial,after)
        record.update(remaining_triangles=remaining,curves=metrics_after,new_pairs=len(after-hits))
        if not remaining:
            if (len(after)<=len(hits) and metrics_after["connected_curve_components"] <=
                base_metrics["connected_curve_components"] and
                metrics_after["length_m"]<=base_metrics["length_m"]):
                out=here/"seat-edge-registered-cap-delta.npz"
                np.savez_compressed(out, row_ids=bank["row_ids"], source_float_delta_m=offset)
                record["provisional_delta"]=out.name
            break
        if set(remaining) <= set(chosen):
            break
    if results and "provisional_delta" in results[-1]:
        break

fixed_before["proposals"]=results
(here/"seat-edge-segment-oriented.json").write_text(json.dumps(fixed_before,indent=2)+"\n")
