def make_candidate(sid,tissue,row,doc,ci,geometry):
    pos=row["positions"]; faces=row["faces"]; local=row["local"]; weights=row["weights"]
    require(1<=row["bc"]<=4,"row exceeds four-slot existing binding format")
    source_manifest=geometry["manifest_rows"][sid]
    expected_member=source_manifest["member_id"]
    require(source_manifest["member_id"]==expected_member,f"stable {sid} source member mismatch")
    require(source_manifest["label"] and source_manifest["body_bindings"],"missing source binding identity")
    union_path=UNION_DIR/f"stable-{sid}-exact-union.json"
    docu=json.loads(union_path.read_text())
    require(docu["source_tissue_sha256"]==EXPECTED_TISS_SHA and docu["source_manifest_sha256"]==EXPECTED_MAN_SHA,
            f"stable {sid} union is not bound to current TISS")
    construction=docu["construction"]; mesh=construction["output_mesh"]
    out_exact=[tuple(rational_pair(v) for v in p) for p in mesh["vertices_m"]]
    out_faces=np.asarray(mesh["triangles"],dtype=np.int64)
    ancestry=docu["ancestry_by_output_face"]
    require(len(out_faces)==len(ancestry)==int(construction["output_face_count"] if "output_face_count" in construction else len(out_faces)),
            "union ancestry/face row mismatch")
    require(all(0<=int(x["source_face"])<len(faces) for x in ancestry),"union source-face ancestry out of range")
    src_unique, inverse=np.unique(pos,axis=0,return_inverse=True)
    qfaces=inverse[faces]
    src_exact=[fraction_tuple(x) for x in src_unique]
    source_coordinate_groups={}
    for vi,p in enumerate(pos):
        source_coordinate_groups.setdefault(fraction_tuple(p),[]).append(vi)
    def source_map(vi):
        m={}
        for slot in range(4):
            ww=Fraction.from_float(float(weights[vi,slot]))
            if ww:
                li=int(local[vi,slot]); require(li<row["bc"],"source active binding index out of range")
                m[li]=m.get(li,Fraction(0))+ww
        return m
    map_by_coord={}
    representative_by_coord={}
    for key,ids in source_coordinate_groups.items():
        maps={map_sig(source_map(i)):source_map(i) for i in ids}
        require(len(maps)==1,f"duplicate source xyz has conflicting route map on stable {sid}")
        map_by_coord[key]=next(iter(maps.values()))
        representative_by_coord[key]=min(ids)
    source_key_by_index={i:tuple(Fraction.from_float(float(x)) for x in p) for i,p in enumerate(src_unique)}
    out_pos=np.empty((len(out_exact),3),dtype="<f4")
    out_weights=np.zeros((len(out_exact),4),dtype="<f4")
    out_local=np.full((len(out_exact),4),np.uint32(0xffffffff),dtype="<u4")
    lift_rows=[]
    new_vertex_indices=[]
    face_parents=[int(x["source_face"]) for x in ancestry]
    incident=[[] for _ in out_exact]
    for fi,f in enumerate(out_faces):
        for vi in f: incident[int(vi)].append(fi)
    max_map_disagreement=Fraction(0)
    max_ancestry_map_sum_error=Fraction(0)
    for vi,p in enumerate(out_exact):
        source_key=tuple(p)
        if source_key in map_by_coord:
            rep=representative_by_coord[source_key]
            out_pos[vi]=pos[rep]
            out_local[vi]=local[rep]
            out_weights[vi]=weights[rep]
            lift_rows.append({"candidate_vertex":vi,"kind":"source_coordinate_preserved",
                              "source_duplicate_group_count":len(source_coordinate_groups[source_key]),
                              "source_representative":rep})
            continue
        out_pos[vi]=np.asarray([round_f32_exact(x) for x in p],dtype="<f4")
        variants={}
        parent_faces=[]
        for fi in incident[vi]:
            sf=face_parents[fi]; parent_faces.append(sf)
            tri_ids=qfaces[sf]
            lamb=barycentric_exact(p,[src_exact[int(k)] for k in tri_ids])
            m={}
            for coeff,src_i in zip(lamb,faces[sf]):
                sm=source_map(int(src_i))
                for li,wgt in sm.items(): m[li]=m.get(li,Fraction(0))+coeff*wgt
            map_sum_error=abs(sum(m.values(),Fraction(0))-1)
            max_ancestry_map_sum_error=max(max_ancestry_map_sum_error,map_sum_error)
            require(float(map_sum_error)<1e-5 and all(v>=0 for v in m.values()),
                    f"stable {sid} ancestry interpolation exceeds inherited composer route-sum tolerance")
            variants[map_sig(m)]=m
        require(bool(variants),f"new stable {sid} union vertex lacks incident parent ancestry")
        maps=[variants[k] for k in sorted(variants)]
        if len(maps)>1:
            l1=max((sum((abs(a.get(k,Fraction(0))-b.get(k,Fraction(0))) for k in range(row["bc"])),Fraction(0))
                    for i,a in enumerate(maps) for b in maps[i+1:]),default=Fraction(0))
            max_map_disagreement=max(max_map_disagreement,l1)
        mean={li:sum((m.get(li,Fraction(0)) for m in maps),Fraction(0))/len(maps)
              for li in range(row["bc"])}
        require(float(abs(sum(mean.values(),Fraction(0))-1))<1e-5 and all(x>=0 for x in mean.values()),
                "consensus map exceeds inherited composer route-sum tolerance")
        out_local[vi,:row["bc"]]=np.arange(row["bc"],dtype="<u4")
        out_weights[vi,:row["bc"]]=np.asarray([np.float32(float(mean[li])) for li in range(row["bc"])],dtype="<f4")
        lift_rows.append({"candidate_vertex":vi,"kind":"equal_mean_distinct_parent_maps",
                          "distinct_map_count":len(maps),"parent_faces":sorted(set(parent_faces)),
                          "max_pairwise_l1_source_map_difference":float(l1) if len(maps)>1 else 0.0,
                          "consensus_exact_local_slots":{str(k):str(v) for k,v in mean.items()},
                          "consensus_f32_local_slots":[float(x) for x in out_weights[vi,:row["bc"]]]})
        new_vertex_indices.append(vi)
    sums=out_weights.astype(np.float64).sum(axis=1)
    require(np.isfinite(out_pos).all() and np.isfinite(out_weights).all() and (out_weights>=0).all(),
            "candidate vertex or route maps are nonfinite/negative")
    require(float(np.max(np.abs(sums-1)))<1e-5,"candidate F32 route weights fail composer sum invariant")
    require(out_faces.min()>=0 and out_faces.max()<len(out_pos),"union candidate face index out of range")
    normals=vertex_normals(out_pos,out_faces)
    vertices6=np.concatenate([out_pos,normals],axis=1).astype("<f4")
    candidate=OUT/f"stable-{sid}-reference-union-row-patch.npz"
    np.savez(candidate,vertices6=vertices6,binding_indices=out_local,weights=out_weights,
             faces=out_faces.astype("<u4"))
    # Candidate source exact self/topology.
    candidate_records,deg=exact_rows(out_pos,out_faces,ci)
    source_records,source_deg=exact_rows(pos,faces,ci)
    require(candidate_records is not None and not deg,"candidate source union is Float32-degenerate")
    candidate_self=ci._audit_pair(candidate_records,candidate_records,same_surface=True)
    source_self=ci._audit_pair(source_records,source_records,same_surface=True)
    top=geometry["analyze_topology"](out_pos.astype(float).tolist(),out_faces.tolist())
    require(not deg and top["closed_oriented_manifold_candidate"],"candidate union topology not closed/oriented manifold")
    require(candidate_self["count"]==0,"candidate source union retains exact self-pairs")
    area_before=area(src_unique,qfaces); area_after=area(out_pos,out_faces)
    source_top=geometry["analyze_topology"](src_unique.astype(float).tolist(),qfaces.tolist())
    source_vols=signed_volumes(top, out_pos, out_faces)
    src_vols=signed_volumes(source_top,src_unique,qfaces)
    report={
      "stable_id":sid,"member_id":expected_member,"label":source_manifest["label"],
      "body_binding_rows":source_manifest["body_bindings"],
      "source_payload_sha256":EXPECTED_TISS_SHA,"source_manifest_sha256":EXPECTED_MAN_SHA,
      "source_row":{"vertex_rows":len(pos),"unique_f32_vertices":len(src_unique),"faces":len(faces),
                    "exact_self_pair_count":source_self["count"],"exact_self_pairs":source_self["triangle_pairs"],
                    "degenerate_face_rows":source_deg,"topology":source_top},
      "source_union":{"output_vertices":len(out_pos),"output_faces":len(out_faces),
                      "exact_self_pair_count":candidate_self["count"],"exact_self_pairs":candidate_self["triangle_pairs"],
                      "degenerate_face_rows":deg,"topology":top,
                      "components_signed_volume_m3":source_vols,"input_components_signed_volume_m3":src_vols,
                      "input_area_m2":area_before,"output_area_m2":area_after,
                      "area_delta_m2":area_after-area_before,"area_delta_relative":area_after/area_before-1,
                      "new_attribute_vertices":len(new_vertex_indices),
                      "ancestry_consensus_max_L1_weight_disagreement":float(max_map_disagreement),
                      "ancestry_exact_route_map_max_abs_sum_error":float(max_ancestry_map_sum_error),
                      "attribute_lift_policy":"For each new union vertex, compute exact barycentric maps through its source-face ancestry, deduplicate distinct exact maps, and take their equal arithmetic mean in the existing row-local binding-index order. Existing source-coordinate vertices preserve their original route arrays. No renormalization, body-binding, route or attachment edit.",
                      "candidate_npz_path":str(candidate),"candidate_npz_sha256":sha(candidate),
                      "npz_fields":["vertices6","binding_indices","weights","faces"],
                      "weights_max_sum_error_f32":float(np.max(np.abs(sums-1))),
                      "weights_renormalized":False,
                      "per_vertex_lift":lift_rows}
    }
    return {"row":row,"positions":out_pos,"weights":out_weights,"local":out_local,
            "faces":out_faces,"vertices6":vertices6,"records":candidate_records,
            "source_records":source_records,"source_self":source_self,"candidate_self":candidate_self,
            "report":report,"candidate_npz":candidate,"topology":top,
            "parents":face_parents,"source_unique":src_unique,"source_qfaces":qfaces}
