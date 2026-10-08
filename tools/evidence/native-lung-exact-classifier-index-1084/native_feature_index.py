"""Native-pose feature indexing for the existing exact lung audit.

Rebuild registered feature coordinates in the pose frame and restrict searches
to common triangle features. Exact intersection and on-edge predicates are unchanged.
"""
from itertools import combinations

def install(base):
    original_classify=base.classify

    def native_lobe_maps(lineage,pose):
        out={}
        for pair,m in lineage["pairs"].items():
            bad=[];vertices=set();edges=set()
            for fa,fb in m["face_pairs"]:
                a,b=pose[pair[0]],pose[pair[1]]
                sa,sb=base.sig(a["v"],a["f"][fa]),base.sig(b["v"],b["f"][fb])
                if sa[0]!=sb[0] or sa[1]!=-sb[1]:
                    bad.append([fa,fb])
                    continue
                ps=sa[0]
                vertices.update(ps)
                edges.update(base.edge(x,y) for x,y in combinations(ps,2))
            feature_count_changed=(len(vertices)!=len(m["shared_vertices"]) or len(edges)!=len(m["shared_edges"]))
            out[pair]={"valid":not bad and not feature_count_changed,
                "face_pairs":m["face_pairs"],"shared_vertices":vertices,"shared_edges":edges,
                "map_count":m["map_count"],"edge_count":m["edge_count"],
                "mismatches":len(bad)+int(feature_count_changed),"examples":bad[:10],
                "feature_coordinate_frame":"actual_accepted_native_world_float32",
                "native_feature_count_changed":feature_count_changed}
        return out

    def classify_lobe_cross(a,b,ra,rb,pts,native_maps):
        pair=(a,b) if a<b else (b,a);m=native_maps.get(pair)
        if m is None:return "unclassified_cross_intersection"
        if not m["valid"]:return "unclassified_current_map_mismatch"
        face_pair=(ra[3],rb[3]) if (a,b)==pair else (rb[3],ra[3])
        if face_pair in m["face_pairs"]:return "exact_reciprocal_face"
        common=set(ra[0])&set(rb[0])
        for v in common:
            if v in m["shared_vertices"] and all(tuple(p)==tuple(v) for p in pts):
                return "source_mapped_shared_vertex"
        for x,y in combinations(common,2):
            e=base.edge(x,y)
            if e in m["shared_edges"] and all(base.pred()._allowed_shared_point(p,set(e)) for p in pts):
                return "source_mapped_shared_edge"
        return "unclassified_cross_intersection"

    def classify(a,b,ra,rb,pts,pose,maps,lobe_maps):
        if b==311:
            return original_classify(b,a,rb,ra,pts,pose,maps,lobe_maps)
        return original_classify(a,b,ra,rb,pts,pose,maps,lobe_maps)

    base.native_lobe_maps=native_lobe_maps
    base.classify_lobe_cross=classify_lobe_cross
    base.classify=classify
    return base
