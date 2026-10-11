def lift_partition_attributes_only(sid,tissue,row,doc,ci,geometry):
    pos=row["positions"]; faces=row["faces"]; local=row["local"]; weights=row["weights"]
    require(row["bc"]>=1,"nonempty source binding table")
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
        active=[li for li in range(row["bc"]) if mean[li]>0]
        require(len(active)<=4,"canonical exact map exceeds existing four nonzero slots")
        out_local[vi,:len(active)]=np.asarray(active,dtype="<u4")
        out_weights[vi,:len(active)]=np.asarray([np.float32(float(mean[li])) for li in active],dtype="<f4")
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
    return {"positions":out_pos,"faces":out_faces,"local":out_local,"weights":out_weights,"parents":face_parents,"lift_rows":lift_rows}
