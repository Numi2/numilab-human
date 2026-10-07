#!/usr/bin/env python3
"""Patch only exact source-owned airway rows 31/32 into a pinned NHANAT5."""
import argparse,copy,hashlib,json,math,shutil,struct,sys
from pathlib import Path
ROOT=Path("/Users/n/numilab-human-airway-junction-source-001")
sys.path.insert(0,str(ROOT/"src"))
from numilab_human import model
from numilab_human.airway_sibling_partition import SOURCE_IDENTITY,SOURCE_MEMBERS,partition_airway_sibling_overlap
H=struct.Struct("<8s5I32s"); R=struct.Struct("<8I"); V=struct.Struct("<6f")
TARGET={31:"FJ2445",32:"FJ2446"}
def sha(b): return hashlib.sha256(b).hexdigest()
def fsha(p): return sha(Path(p).read_bytes())
def u32(v): return struct.pack(f"<{len(v)}I",*v)
def parse(raw):
    magic,abi,n,nv,ni,reg,src=H.unpack_from(raw); vo=H.size+n*R.size; io=vo+nv*24
    if magic!=b"NHANAT1\0" or abi!=5 or len(raw)!=io+ni*4: raise ValueError("invalid NHANAT5 ABI/length")
    ix=struct.unpack_from(f"<{ni}I",raw,io); rows=[]; vv=ii=0; seen=set()
    for j in range(n):
        body,fv,vc,fi,ic,sid,layer,flags=R.unpack_from(raw,H.size+j*R.size)
        if sid in seen or fv!=vv or fi!=ii or ic%3 or fv+vc>nv or fi+ic>ni: raise ValueError(f"invalid row {sid}")
        seen.add(sid); local=[ix[k]-fv for k in range(fi,fi+ic)]
        if any(k<0 or k>=vc for k in local): raise ValueError(f"bad index at {sid}")
        rows.append(dict(body=body,fv=fv,vc=vc,fi=fi,ic=ic,sid=sid,layer=layer,flags=flags,
            vb=raw[vo+fv*24:vo+(fv+vc)*24],local=local))
        vv+=vc;ii+=ic
    if vv!=nv or ii!=ni: raise ValueError("surface rows do not cover payload")
    return (magic,abi,n,nv,ni,reg,src),rows
def content(rows):
    h=hashlib.sha256()
    for x in rows:
        ib=u32(x["local"]); h.update(struct.pack("<III",x["sid"],len(x["vb"]),len(ib)));h.update(x["vb"]);h.update(ib)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("base-payload","base-receipt","sources","registration","myosim-artifact","output-dir"): p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--surface-map",type=Path,default=ROOT/"config/bodyparts3d-myosim-torso-anatomy-map.v1.json")
    a=p.parse_args(); base=a.base_payload.resolve(); rp=a.base_receipt.resolve(); out=a.output_dir.resolve()
    if out.exists(): raise FileExistsError(out)
    receipt=model.read_json(rp); pin=receipt["payload"]
    if pin["path"]!=str(base) or fsha(base)!=pin["sha256"]: raise ValueError("base payload does not match receipt")
    raw=base.read_bytes(); head,rows=parse(raw); magic,abi,n,nv,ni,reg,src=head; source=src.hex()
    if source!=pin["source_sha256"] or reg!=int(pin["registration_fingerprint32"],16): raise ValueError("payload source/frame pin mismatch")
    if fsha(a.registration)!=receipt["provenance"]["bodyparts_registration_sha256"]: raise ValueError("registration hash mismatch")
    mappath=a.surface_map.resolve(); sm=model.read_json(mappath); mapsha=fsha(mappath)
    old={x["sid"]:x for x in rows if x["sid"] in TARGET}
    if set(old)!=set(TARGET) or any(old[s]["body"]!=20 or old[s]["layer"]!=4 for s in TARGET): raise ValueError("base 31/32 identity mismatch")
    specs={e["member_id"]:e for e in sm["entries"] if e.get("member_id") in SOURCE_MEMBERS}
    if set(specs)!=set(SOURCE_MEMBERS): raise ValueError("map lacks exact sibling rows")
    source_relations=model._bodyparts_source_element_relation_names(a.sources.resolve(),"is_a")
    inputs={}; objmeta={}
    for m in SOURCE_MEMBERS:
        archive,name,obj=model._bodyparts_obj_member(a.sources.resolve(),"is_a",m); verts,tris=model._bodyparts_obj_triangles(obj,name); sh=sha(obj)
        ident=SOURCE_IDENTITY[m]
        if sh!=ident["member_sha256"] or (specs[m]["concept_id"],specs[m]["source_name"],specs[m]["layer"])!=(ident["concept_id"],ident["label"],"airway") or (ident["concept_id"],ident["label"],m) not in source_relations: raise ValueError(f"{m} source identity or is_a relation mismatch")
        inputs[m]=dict(vertices_mm=verts,triangles=tris,source_sha256=sh); objmeta[m]=dict(archive=archive.name,member=name,sha256=sh,source_face_count=len(tris))
    part=partition_airway_sibling_overlap(inputs); proof=part["proof"]; cert=proof["certificate"]
    json.dumps(proof,sort_keys=True,allow_nan=False)
    if not (cert["source_face_coverage_exact"] and cert["child_interiors_uncut"] and cert["independent_classification_exact"] and cert["source_union_preserved"] and cert["source_exclusive_regions_preserved"] and cert["union"]["topology"]["closed"] and cert["union"]["topology"]["face_components"]==1): raise ValueError("source partition certificate failed")
    registration=model.read_json(a.registration); gt,gq,gs=model._bodyparts_visual_local_pose(registration["coordinate_system"]["global_source_mm_to_myosim_world_m"],"airway source")
    grot=model._myosim_matrix_from_quaternion_xyzw(gq); bodies,_,_=model._myosim_surface_route_context(a.myosim_artifact.resolve(),source)
    replacements={}
    for m,sid in (("FJ2445",31),("FJ2446",32)):
        body=bodies[specs[m]["myosim_body"]]; bi=body["core_body_index"]; pos=body["default_com_position_world_m"]; quat=body["default_inertial_quaternion_world_xyzw"]
        mesh=part["meshes"][m]; verts=mesh["vertices_mm"]; tris=mesh["triangles"]; norms=model._bodyparts_vertex_normals(verts,tris,m)
        world=model._bodyparts_source_mm_to_body_world(verts,[0.,0.,0.],[0.,0.,0.,1.],gt,gq,gs)
        stored=model._bodyparts_world_to_body_stored_m(world,pos,quat,[0.,0.,0.],[0.,0.,0.,1.],1.,m)
        inv=model._matrix_transpose(model._myosim_matrix_from_quaternion_xyzw(quat)); ns=[]
        for nrm in norms:
            wn=model._bodyparts_unit_vector(model._myosim_matrix_vector(grot,list(nrm)),m+" world normal")
            ns.append(model._bodyparts_unit_vector(model._myosim_matrix_vector(inv,wn),m+" body normal"))
        vb=b"".join(V.pack(*(tuple(v)+tuple(nrm))) for v,nrm in zip(stored,ns,strict=True))
        lineage=proof["face_lineage"][m]
        if len(tris)!=lineage["emitted_face_count"] or len(mesh["source_face_indices"])!=len(tris): raise ValueError("face lineage mismatch")
        replacements[sid]=dict(body=bi,layer=4,flags=0,vc=len(verts),vb=vb,local=[i for t in tris for i in t],member=m,body_name=specs[m]["myosim_body"],face_count=len(tris))
    common=receipt["provenance"]["cardiac_geometry_binding"]["common_field"]
    out.mkdir(parents=True); copied={}
    for key in ("map","polynomials","domain_boxes"):
        d=common[key]; srcp=rp.parent/d["path"]
        if fsha(srcp)!=d["sha256"]: raise ValueError("base common-field asset hash mismatch: "+key)
        dst=out/Path(d["path"]).name; shutil.copyfile(srcp,dst)
        if fsha(dst)!=d["sha256"]: raise ValueError("common-field copy mismatch: "+key)
        copied[key]=dict(file=dst.name,sha256=fsha(dst),bytes=dst.stat().st_size)
    md=(out/Path(common["map"]["path"]).name).read_bytes(); mc=common["map"]["record_count"]; stride=common["map"]["record_stride_bytes"]
    if stride!=112 or len(md)!=mc*stride: raise ValueError("common-field map size/stride mismatch")
    for row in struct.iter_unpack("<28f",md):
        if not all(math.isfinite(v) for v in row): raise ValueError("nonfinite common-field coefficient")
        if any(struct.unpack("<I",struct.pack("<f",row[j]))[0]!=0 for j in range(3,28,4)): raise ValueError("common-field W lane not +0")
    active=common["vertex_ranges"]; passive=common["passive_attachment_ranges"]
    if any(x["stable_id"] in TARGET for x in active+passive): raise ValueError("airway IDs have unexpected common-field rows")
    ranges=[i for x in active+passive for i in range(x["first_vertex"],x["first_vertex"]+x["vertex_count"])]
    if len(ranges)!=mc or len(set(ranges))!=mc or sorted(ranges)!=list(range(mc)): raise ValueError("common-field map ranges do not tile all records")
    mapproof=dict(record_count=mc,record_stride_bytes=stride,map_sha256=common["map"]["sha256"],coefficient_tuple_sha256=sha(md),
        finite_coefficients=True,all_float4_w_lanes_exact_positive_zero=True,active_volume_surface_ids=[x["stable_id"] for x in active],
        passive_surface_count=len(passive),passive_map_record_count=sum(x["vertex_count"] for x in passive),
        active_and_passive_ranges_cover_all_map_records=True,target_ids_without_map_rows=[31,32],
        range_basis=common.get("vertex_range_interpretation"),assets=copied,
        interpretation="Ranges index concatenated common-map records, not global NHANAT vertex offsets. No coefficients are added for target airway IDs.")
    recs=[];vparts=[];iparts=[];nextv=nexti=0
    for x in rows:
        y=replacements.get(x["sid"],x); vc=y.get("vc",x["vc"]); vb=y["vb"]; local=y["local"]
        if y["body"]!=x["body"] or y["layer"]!=x["layer"] or len(vb)!=vc*24 or len(local)%3 or any(i<0 or i>=vc for i in local): raise ValueError(f"invalid composed surface {x['sid']}")
        recs.append(R.pack(y["body"],nextv,vc,nexti,len(local),x["sid"],y["layer"],y.get("flags",x["flags"])))
        vparts.append(vb);iparts.append(u32([nextv+i for i in local]));nextv+=vc;nexti+=len(local)
    output=H.pack(magic,abi,n,nextv,nexti,reg,src)+b"".join(recs)+b"".join(vparts)+b"".join(iparts); payload=out/"resting-thorax.nhanatomy";payload.write_bytes(output); ph=sha(output)
    _,newrows=parse(output); before=[x for x in rows if x["sid"] not in TARGET]; after=[x for x in newrows if x["sid"] not in TARGET]
    if len(before)!=n-2 or content(before)!=content(after): raise ValueError("non-target NHA geometry/index bytes changed")
    passive_before=json.dumps(receipt["functional_bindings"]["passive_viscera_geometry_binding"],sort_keys=True,separators=(",",":"))
    cand=copy.deepcopy(receipt);cand["payload"].update(path=str(payload),sha256=ph,vertex_count=nextv,index_count=nexti)
    cand["functional_bindings"]["anatomy_payload_sha256"]=ph;cand["provenance"]["bodyparts3d_torso_owner_map_sha256"]=mapsha
    ccommon=cand["provenance"]["cardiac_geometry_binding"]["common_field"];ccommon["anatomy_payload_sha256"]=ph
    proofsha=sha(json.dumps(proof,sort_keys=True,separators=(",",":"),allow_nan=False).encode())
    cand["provenance"]["airway_sibling_overlap_partition"]=dict(schema="numi.human.airway-sibling-solid-proxy-partition-evidence.v1",
        qualification="exact source geometry candidate; not lumen or physiological qualification",base_payload_path=str(base),base_payload_sha256=pin["sha256"],
        output_payload_path=str(payload),output_payload_sha256=ph,base_receipt_path=str(rp),base_receipt_sha256=fsha(rp),
        surface_map_path=str(mappath),surface_map_sha256=mapsha,registration_sha256=fsha(a.registration),source_archive_sha256=source,
        target_stable_ids=[31,32],source_identity=proof["source_identity"],source_geometry_identity=proof["source_geometry_identity"],
        face_lineage=proof["face_lineage"],exact_source_certificate=cert,exact_source_certificate_sha256=proofsha,
        members={str(s):dict(source_member=replacements[s]["member"],myosim_body=replacements[s]["body_name"],source_obj=objmeta[replacements[s]["member"]],
            emitted_vertex_count=replacements[s]["vc"],emitted_face_count=replacements[s]["face_count"],emitted_index_count=len(replacements[s]["local"])) for s in (31,32)},
        unchanged_surface_payloads=dict(surface_count=n-2,vertex_bytes_and_local_indices_exact=True,content_sha256=content(after)),
        common_cardiac_field_preservation=mapproof,source_solid_interpretation_boundary=proof["geometry_interpretation"],
        native_emission_boundary="Exact certificate is for source binary64 geometry before transform/float32 payload emission; no clinical lumen is asserted.",
        binding_boundary="Receipt 814 passive-viscera binding copied unchanged; no airway physical or physiological binding added.")
    if json.dumps(cand["functional_bindings"]["passive_viscera_geometry_binding"],sort_keys=True,separators=(",",":"))!=passive_before: raise ValueError("passive binding changed")
    rout=out/"resting-anatomy-receipt.json";rout.write_text(json.dumps(cand,indent=2,sort_keys=True)+"\n")
    print(json.dumps(dict(payload=str(payload),payload_sha256=ph,receipt=str(rout),receipt_sha256=fsha(rout),surface_count=n,vertex_count=nextv,index_count=nexti,
        rows={str(s):dict(vertices=replacements[s]["vc"],triangles=replacements[s]["face_count"]) for s in (31,32)},
        untouched_surface_count=n-2,untouched_surface_content_sha256=content(after),common_map_sha256=mapproof["map_sha256"],
        common_map_coefficients_unchanged=True,exact_source_certificate_sha256=proofsha,native_preflight_or_simulation_run=False),indent=2,sort_keys=True))
if __name__=="__main__": main()
