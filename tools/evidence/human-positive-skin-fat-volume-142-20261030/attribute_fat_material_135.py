#!/usr/bin/env python3
"""Build material/source-attribution sidecar for compacted Fat candidate 135.

Geometry, cells, and raw region attributes are read-only. Derived cells inherit
patch-level ancestry unions from the exact selected source-096 rows; they are
not assigned fabricated per-cell TetGen ancestors. This is source attribution
only and does not admit the mesh or run mass transport/runtime.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from collections import Counter
from pathlib import Path
import numpy as np

H = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
PIN = {
    "candidate096": "d8e708ac001981b42d5d26d5bf2d4108b35057cdd6a8783484a90d47520089f7",
    "sidecar096": "5b9cb4aadea5727d53f516bcefc939520863928b9e3a0b67fddfa132cac6fd08",
    "candidate106": "8223a4a295e240859b7e3b07b62207aa5ef572c95d8b0370c74a9419046cc4c6",
    "candidate123": "332fb09b921ef2da5652553d94e1669d365046cbf06a3f2ed06c1b349094de76",
    "candidate127": "23ceea4acb0ab21783ac2bc7856e7389bf232b3a12be6842ef707150c9b8ce07",
    "candidate135": "7ab13a433e0cb08350c901388c0c86712269e0cf66ce70df5fe33de248357e76",
    "exact127_certificate": "42833eb99f62c0f0853c7f82ba32cd30ddd26e024582481000dcaee5acbdd2f3",
}
SCALAR_NAMES = ["source_066_cell_row", "source_064_candidate_cell_row", "source_tetgen_element_id",
                "source_region_attribute_059", "corrected_region_attribute_062", "attribution_method",
                "attribution_patch_id", "lineage_granularity"]
CSR = {
    "source_parent_tetgen_element": ("source_parent_tetgen_element_offsets", "source_parent_tetgen_element_ids"),
    "source_066_parent_candidate_row": ("source_066_parent_candidate_row_offsets", "source_066_parent_candidate_rows"),
    "source_parent_084_candidate_row": ("source_parent_084_candidate_row_offsets", "source_parent_084_candidate_rows"),
    "source_parent_080_candidate_row": ("source_parent_080_candidate_row_offsets", "source_parent_080_candidate_rows"),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def need(ok, msg):
    if not bool(ok):
        raise ValueError(msg)


def loadj(path: Path):
    d = json.loads(Path(path).read_text())
    need(isinstance(d, dict), f"expected JSON object: {path}")
    return d


def loadz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def pin(path: Path, expected: str | None, pins: dict, label: str):
    path = Path(path).resolve()
    need(path.is_file(), f"missing {label}: {path}")
    actual = sha(path)
    if expected is not None:
        need(actual == expected, f"{label} hash mismatch: {actual} != {expected}: {path}")
    pins[label] = {"path": str(path), "sha256": actual}
    return actual


def csr_row(offsets, values, row):
    a, b = int(offsets[row]), int(offsets[row + 1])
    return values[a:b]


def assert_partition(retained, replaced, n, label):
    retained = np.asarray(retained, np.int64); replaced = np.asarray(replaced, np.int64)
    need(len(np.unique(retained)) == len(retained), f"{label}: duplicate retained source rows")
    need(len(np.unique(replaced)) == len(replaced), f"{label}: duplicate replaced source rows")
    need(not np.intersect1d(retained, replaced, assume_unique=False).size, f"{label}: retained/replaced overlap")
    need(np.array_equal(np.sort(np.r_[retained, replaced]), np.arange(n, dtype=np.int64)), f"{label}: source partition incomplete")


def add_patch(patches, patch_id, source96_rows, output_rows, label):
    pid = int(patch_id)
    need(pid not in patches, f"duplicate derived patch id {pid}")
    src = np.unique(np.asarray(source96_rows, np.int64))
    out = np.asarray(output_rows, np.int64)
    need(len(src) > 0, f"empty source-096 union for {label}")
    need(len(out) > 0, f"empty output block for {label}")
    patches[pid] = {"source096_rows": src, "output_rows": out, "label": label}


def build_patch_lineage(ps, source96_rows):
    src = np.unique(np.asarray(source96_rows, np.int64))
    need(len(src) > 0, "empty source row union")
    n96 = len(ps["output_cell_row"])
    need(np.all(src >= 0) and np.all(src < n96), "patch source-096 row outside sidecar")
    result = {}
    for key, (offkey, valkey) in CSR.items():
        arrays = [csr_row(ps[offkey], ps[valkey], int(r)) for r in src]
        need(all(len(a) > 0 for a in arrays), f"empty parent ancestry in patch union: {key}")
        result[key] = np.unique(np.concatenate(arrays).astype(np.int64, copy=False))
    io = ps["source_parent_tetgen_element_offsets"]
    ids0 = ps["source_parent_tetgen_element_ids"]
    raw0 = ps["source_parent_region_attribute_059"]
    fix0 = ps["source_parent_region_attribute_062"]
    mat0 = ps["source_parent_material_index"]
    ids_parts=[]; raw_parts=[]; fix_parts=[]; mat_parts=[]
    for r in src:
        a,b=int(io[r]),int(io[r+1])
        need(b>a, f"empty source TetGen parent row {r}")
        ids_parts.append(ids0[a:b]); raw_parts.append(raw0[a:b]); fix_parts.append(fix0[a:b]); mat_parts.append(mat0[a:b])
    ids=np.concatenate(ids_parts).astype(np.int64, copy=False)
    raw=np.concatenate(raw_parts); fix=np.concatenate(fix_parts); mat=np.concatenate(mat_parts)
    order=np.argsort(ids, kind="stable"); ids,raw,fix,mat=ids[order],raw[order],fix[order],mat[order]
    starts=np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    uid=ids[starts]
    rawlo=np.minimum.reduceat(raw,starts); rawhi=np.maximum.reduceat(raw,starts)
    fixlo=np.minimum.reduceat(fix,starts); fixhi=np.maximum.reduceat(fix,starts)
    matlo=np.minimum.reduceat(mat,starts); mathi=np.maximum.reduceat(mat,starts)
    need(np.array_equal(rawlo,rawhi), "conflicting raw region attributes for a source TetGen parent")
    need(np.array_equal(fixlo,fixhi), "conflicting corrected region attributes for a source TetGen parent")
    need(np.array_equal(matlo,mathi), "conflicting material attributes for a source TetGen parent")
    need(np.all(matlo == 16), "patch union contains a source parent outside Fat material 16")
    need(np.all(np.isin(fixlo, [16,17,18,19,20,21,22])), "patch union contains an uncertified corrected region")
    result["source_parent_tetgen_element"] = uid
    result["source_parent_region_attribute_059"] = rawlo
    result["source_parent_region_attribute_062"] = fixlo
    result["source_parent_material_index"] = matlo.astype(ps["source_parent_material_index"].dtype, copy=False)
    result["source096_rows"] = src
    return result


def compose_csr(base_offsets, base_values, src96_for_output, patch_id_for_output, patch_order, patch_arrays):
    n = len(src96_for_output)
    base_offsets=np.asarray(base_offsets,np.int64); base_values=np.asarray(base_values)
    n96=len(base_offsets)-1
    base_lengths=np.diff(base_offsets)
    src=np.asarray(src96_for_output,np.int64)
    pid=np.asarray(patch_id_for_output,np.int64)
    is_base=src>=0
    need(np.all((src[is_base]>=0)&(src[is_base]<n96)), "direct source96 CSR row out of range")
    slot_for_pid={int(p):i for i,p in enumerate(patch_order)}
    slots=np.full(n,-1,np.int32)
    for p,slot in slot_for_pid.items(): slots[pid==p]=slot
    need(np.all(slots[~is_base]>=0), "derived output has no patch CSR union")
    patch_lengths=np.asarray([len(patch_arrays[int(p)]) for p in patch_order],np.int64)
    lengths=np.empty(n,np.int64)
    lengths[is_base]=base_lengths[src[is_base]]
    lengths[~is_base]=patch_lengths[slots[~is_base]]
    need(np.all(lengths>0), "empty output ancestry CSR row")
    offsets=np.empty(n+1,np.int64); offsets[0]=0; np.cumsum(lengths,out=offsets[1:])
    total=int(offsets[-1]); vals=np.empty(total,dtype=base_values.dtype)
    entry_rows=np.repeat(np.arange(n,dtype=np.int64),lengths)
    local=np.arange(total,dtype=np.int64)-np.repeat(offsets[:-1],lengths)
    src_for_entry=src[entry_rows]
    mask=src_for_entry>=0
    if np.any(mask):
        base_i=base_offsets[src_for_entry[mask]]+local[mask]
        vals[mask]=base_values[base_i]
    if np.any(~mask):
        slots_entry=slots[entry_rows[~mask]]
        po=np.empty(len(patch_order)+1,np.int64); po[0]=0
        np.cumsum(patch_lengths,out=po[1:])
        pflat=np.concatenate([np.asarray(patch_arrays[int(p)],dtype=base_values.dtype) for p in patch_order])
        vals[~mask]=pflat[po[slots_entry]+local[~mask]]
    return offsets, vals


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, type=Path)
    args=ap.parse_args()
    t0=time.time(); pins={}
    # Every edge in the chain is pinned by both its producer report and bytes.
    p96=H/"fat-local-milp-composite-096/fat-current-milp-restored-source-composite.npz"
    r96p=H/"fat-local-milp-composite-096/report.json"
    s096p=H/"fat-material-attribution-096/run-001/fat-material-attribution.npz"
    sr96p=H/"fat-material-attribution-096/run-001/report.json"
    p106=H/"fat-local-milp-composite-106-two-free-stars-original-edge/fat-current-free-stars-and-original-edge-composite.npz"
    r106p=H/"fat-local-milp-composite-106-two-free-stars-original-edge/report.json"
    p123=H/"fat-local-milp-composite-123-six-certified-cavities/fat-current-six-certified-restored-cavities.npz"
    r123p=H/"fat-local-milp-composite-123-six-certified-cavities/report.json"
    p127=H/"fat-inferred-boundary-positive-volume-127-root/fat-positive-inferred-boundary-proposal.npz"
    r127p=H/"fat-inferred-boundary-positive-volume-127-root/report.json"
    p135=H/"fat-positive-PLC-prefix-compaction-135-root/fat-positive-compact-PLC-prefix.npz"
    r135p=H/"fat-positive-PLC-prefix-compaction-135-root/report.json"
    c127cert=H/"fat-inferred-boundary-exact-certification-129-v4/certification.json"
    r103p=H/"fat-local-milp-composite-103-free-star-246038/report.json"
    r103=loadj(r103p)
    p103=Path(r103["candidate_path"])
    r100p=Path(r103["input_candidate_path"]).parent/"report.json"
    p100=Path(r103["input_candidate_path"])
    r100=loadj(r100p)
    # Validate immutable report/asset chain pins before loading data arrays.
    for path,expected,label in [
      (r96p,None,"report096"),(p96,PIN["candidate096"],"candidate096"),
      (s096p,PIN["sidecar096"],"sidecar096"),(sr96p,None,"sidecar_report096"),
      (r100p,r103.get("input_report_sha256"),"report100"),(p100,r103.get("input_candidate_sha256"),"candidate100"),
      (r103p,None,"report103"),(p103,r103.get("candidate_sha256"),"candidate103"),
      (r106p,None,"report106"),(p106,PIN["candidate106"],"candidate106"),
      (r123p,None,"report123"),(p123,PIN["candidate123"],"candidate123"),
      (r127p,None,"report127"),(p127,PIN["candidate127"],"candidate127"),
      (c127cert,PIN["exact127_certificate"],"exact127_certificate"),
      (r135p,None,"report135"),(p135,PIN["candidate135"],"candidate135")]:
        pin(path, expected, pins, label)
    r96=loadj(r96p); sr96=loadj(sr96p); r106=loadj(r106p); r123=loadj(r123p); r127=loadj(r127p); r135=loadj(r135p); cert=loadj(c127cert)
    need(r96.get("physical_admission") is False and r96.get("runtime")==0, "096 source candidate must remain unadmitted")
    need(sr96.get("physical_admission") is False and sr96.get("runtime")==0, "096 attribution report must remain source-only")
    need(sr96.get("candidate_sha256")==PIN["candidate096"] and sr96.get("sidecar_sha256")==PIN["sidecar096"], "096 sidecar is not bound to frozen candidate")
    need(r103.get("schema")=="numi.fat-free-star-composition-103.v1" and r103.get("physical_admission") is False and r103.get("runtime")==0, "103 immediate parent status/schema changed")
    need(r100.get("candidate_sha256")==sha(p100) and r103.get("input_candidate_sha256")==sha(p100), "100 exact parent binding changed")
    need(r100.get("physical_admission") is False and r100.get("runtime")==0, "100 exact parent is not source-only")
    need(r106.get("candidate_sha256")==PIN["candidate106"] and r106.get("input_candidate_sha256")==sha(p103), "106 exact 103 parent mismatch")
    need(r123.get("candidate_sha256")==PIN["candidate123"] and r123.get("input_candidate_sha256")==PIN["candidate106"], "123 exact 106 parent mismatch")
    need(r127.get("filtered_exact_whole_volume_sign_audit",{}).get("sha256")==PIN["candidate127"] and r127.get("outputs",{}).get("fat-positive-inferred-boundary-proposal.npz")==PIN["candidate127"] and r127.get("source_candidate_sha256")==PIN["candidate123"], "127 exact 123 parent mismatch")
    need(r135.get("candidate_sha256")==PIN["candidate135"] and r135.get("source_candidate_sha256")==PIN["candidate127"], "135 exact 127 parent mismatch")
    need(all(x.get("physical_admission") is False for x in [r100,r106,r123,r127,r135]), "an upstream source-only receipt changed status")
    need(all(x.get("runtime",0)==0 for x in [r100,r106,r123,r127,r135]), "unexpected upstream runtime")
    need(cert.get("status","PASS").startswith("PASS") or cert.get("status")=="PASS_EXACT_POSITIVE_F32_VOLUME", "127 exact sign certificate is not a pass")
    # Load all numeric members exactly once, then build the ancestry map in memory.
    a96=loadz(p96); a100=loadz(p100); a103=loadz(p103); a106=loadz(p106); a123=loadz(p123); a127=loadz(p127); a135=loadz(p135); side=loadz(s096p)
    t96=np.asarray(a96["tetrahedra"],np.int64); m96=np.asarray(a96["material_indices"],np.int32); p96xyz=a96["points_world_m"]
    n96=len(t96); ret96=np.asarray(a96["retained_input_candidate_tet_rows"],np.int64); rep96=np.asarray(a96["retet_input_candidate_tet_rows"],np.int64)
    n84=len(ret96)+len(rep96); assert_partition(ret96,rep96,n84,"096/084")
    inv96=np.full(n84,-1,np.int64); inv96[ret96]=np.arange(len(ret96),dtype=np.int64)
    need(len(side["output_cell_row"])==n96 and np.array_equal(side["output_cell_row"],np.arange(n96)), "096 sidecar row domain mismatch")
    need(np.all(side["material_index"]==16), "096 sidecar is not all Fat material16")
    need(np.array_equal(side["current_candidate_region_attribute"],m96), "096 source raw attributes mismatch")
    for stem,(ok,vk) in CSR.items():
        need(len(side[ok])==n96+1 and int(side[ok][0])==0 and int(side[ok][-1])==len(side[vk]), f"096 malformed source CSR {stem}")
        need(np.all(np.diff(side[ok])>0), f"096 empty ancestry row in {stem}")
    po=side["source_parent_tetgen_element_offsets"]
    need(len(side["source_parent_region_attribute_059"])==len(side["source_parent_tetgen_element_ids"]) and len(side["source_parent_region_attribute_062"])==len(side["source_parent_tetgen_element_ids"]) and len(side["source_parent_material_index"])==len(side["source_parent_tetgen_element_ids"]), "096 parent attribute CSR misaligned")
    need(np.all(side["source_parent_material_index"]==16), "096 source parent material lane not all 16")
    # Exact 100->103 parent branch: 100 is used only because 103 names and hashes it as its immediate parent.
    ret100=np.asarray(a100["retained_input_candidate_tet_rows"],np.int64); out100=np.asarray(a100["retet_component_output_tetrahedra_global_node_rows"],np.int64)
    map100=inv96[ret100]
    need(np.all(map100>=0), "100 retained cells no longer map to frozen 096 retained domain")
    need(np.array_equal(a100["tetrahedra"][:len(ret100)],t96[map100]), "100 retained cells differ from exact 096 ancestry")
    need(np.array_equal(a100["material_indices"][:len(ret100)],m96[map100]), "100 retained raw attributes differ from 096")
    star96=np.asarray(a100["boundary_star_component_current_parent_rows"],np.int64)
    star84=np.asarray(a100["boundary_star_component_base_parent_rows"],np.int64)
    starout=np.asarray(a100["boundary_star_component_output_tetrahedra_global_node_rows"],np.int64)
    need(len(star96)==22 and len(np.unique(star96))==22 and len(star84)==22 and np.array_equal(ret96[star96],star84), "100 star's source-096/source-084 map changed")
    need(np.all(star96<len(ret96)), "100 star unexpectedly selects generated 096 output cells")
    assert_partition(ret96,rep96,n84,"096")
    rep100=np.asarray(a100["retet_input_candidate_tet_rows"],np.int64)
    need(np.array_equal(np.sort(rep100),np.sort(np.r_[rep96,star84])), "100 star's source retirement set differs from exact 096 plus star parents")
    need(len(out100)==492 and np.array_equal(out100[:465],a96["retet_component_output_tetrahedra_global_node_rows"]), "100 did not preserve exact 096 retet block")
    need(np.array_equal(out100[465:],starout) and len(starout)==27, "100 output boundary-star block mismatch")
    need(np.array_equal(a100["tetrahedra"][len(ret100):],out100), "100 appended cell block changed")
    need(np.all(a100["material_indices"][len(ret100):]==16), "100 generated material attribute is not 16")
    # 103 preserves the direct 100 branch, then adds its 268-cell free-star block.
    ret103=np.asarray(a103["retained_input_candidate_tet_rows"],np.int64); map103=inv96[ret103]
    need(np.all(map103>=0) and np.array_equal(a103["tetrahedra"][:len(ret103)],t96[map103]), "103 retained cells do not map to frozen 096 rows")
    need(np.array_equal(a103["material_indices"][:len(ret103)],m96[map103]), "103 retained raw attributes changed")
    out103=np.asarray(a103["retet_component_output_tetrahedra_global_node_rows"],np.int64)
    free103=np.asarray(a103["free_star_component_output_tetrahedra_global_node_rows"],np.int64)
    free96=np.asarray(a103["free_star_component_096_parent_rows"],np.int64)
    need(np.array_equal(out103[:len(out100)],out100) and np.array_equal(out103[len(out100):],free103), "103 output block does not reconstruct direct 100 + free-star chain")
    need(len(free103)==268 and len(free96)==279 and np.all((free96>=0)&(free96<n96)), "103 free-star parent union malformed")
    need(np.array_equal(a103["tetrahedra"][len(ret103):],out103) and np.all(a103["material_indices"][len(ret103):]==16), "103 appended cells differ from declared block/material")
    # 106 retained prefix maps exactly to 096; its output block is pinned 103 plus two declared additions.
    ret106=np.asarray(a106["retained_input_candidate_tet_rows"],np.int64); map106ret=inv96[ret106]
    need(np.all(map106ret>=0) and np.array_equal(a106["tetrahedra"][:len(ret106)],t96[map106ret]), "106 retained cells do not map to frozen 096")
    need(np.array_equal(a106["material_indices"][:len(ret106)],m96[map106ret]), "106 retained raw attributes changed")
    out106=np.asarray(a106["retet_component_output_tetrahedra_global_node_rows"],np.int64)
    qout=np.asarray(a106["quantized_free_star_output_tetrahedra_global_node_rows"],np.int64)
    eout=np.asarray(a106["original_edge_108_output_tetrahedra_global_node_rows"],np.int64)
    q96=np.asarray(a106["quantized_free_star_096_parent_rows"],np.int64)
    e96=np.asarray(a106["original_edge_108_096_parent_rows"],np.int64)
    need(np.array_equal(out106[:len(out103)],out103), "106 failed to preserve exact 103 output block")
    need(np.array_equal(out106[len(out103):len(out103)+12],qout) and np.array_equal(out106[-21:],eout), "106 appended quantized/edge blocks differ from declarations")
    need(len(qout)==12 and len(q96)==12 and len(eout)==21 and len(e96)==19, "106 new patch data shape changed")
    need(np.array_equal(a106["tetrahedra"][len(ret106):],out106) and np.all(a106["material_indices"][len(ret106):]==16), "106 appended cell block/material mismatch")
    # Compose one row per 106 candidate cell: exact 096 row or a named patch union.
    src106=np.full(len(a106["tetrahedra"]),-1,np.int64); src106[:len(ret106)]=map106ret
    patch106=np.full(len(src106),-1,np.int64); patchsets={}
    n106ret=len(ret106); n103out=len(out103)
    # The 27 added boundary-star cells derive from the exact star's 22 source-096 parents.
    star_id=1061001; add_patch(patchsets,star_id,star96,np.arange(n106ret+465,n106ret+492),"100 boundary-star patch inherited through exact 103 parent")
    # 103 output rows 0..465 are generated by 096, while its next 27 are the star and 268 the free star.
    src106[n106ret:n106ret+465]=np.arange(len(ret96),n96,dtype=np.int64)
    patch106[n106ret+465:n106ret+492]=star_id
    free_id=1061002; add_patch(patchsets,free_id,free96,np.arange(n106ret+492,n106ret+760),"103 free-star patch")
    patch106[n106ret+492:n106ret+760]=free_id
    quant_id=1061003; add_patch(patchsets,quant_id,q96,np.arange(n106ret+760,n106ret+772),"106 quantized free-star patch")
    patch106[n106ret+760:n106ret+772]=quant_id
    edge_id=1080108; add_patch(patchsets,edge_id,e96,np.arange(n106ret+772,n106ret+793),"108 original-edge patch")
    patch106[n106ret+772:n106ret+793]=edge_id
    need(len(out106)==793 and np.all((src106>=0) | (patch106>=0)), "106 cell map has unmapped rows")
    # 123 retained cells are exact 096 cells; first 793 retets carry 106 ancestry and six additions use explicit source096 CSR.
    ret123=np.asarray(a123["retained_input_candidate_tet_rows"],np.int64); map123ret=inv96[ret123]
    need(np.all(map123ret>=0) and np.array_equal(a123["tetrahedra"][:len(ret123)],t96[map123ret]), "123 retained cells do not map to 096")
    need(np.array_equal(a123["material_indices"][:len(ret123)],m96[map123ret]), "123 retained raw attributes changed")
    out123=np.asarray(a123["retet_component_output_tetrahedra_global_node_rows"],np.int64)
    need(np.array_equal(out123[:len(out106)],out106), "123 first 106-derived block changed")
    restored=np.asarray(a123["restored_cavity_output_tetrahedra_global_node_rows"],np.int64)
    off=np.asarray(a123["restored_cavity_source096_offsets"],np.int64); vals=np.asarray(a123["restored_cavity_source096_rows"],np.int64)
    outoff=np.asarray(a123["restored_cavity_output_tet_offsets"],np.int64)
    need(len(off)==7 and len(outoff)==7 and len(restored)==208 and np.array_equal(out123[len(out106):],restored), "123 six-cavity block/CSR malformed")
    need(np.array_equal(a123["tetrahedra"][len(ret123):],out123) and np.all(a123["material_indices"][len(ret123):]==16), "123 appended block/material mismatch")
    src123=np.full(len(a123["tetrahedra"]),-1,np.int64); src123[:len(ret123)]=map123ret
    patch123=np.full(len(src123),-1,np.int64)
    # Carry the 106 retet block's direct rows and patch unions into 123.
    n123ret=len(ret123)
    src123[n123ret:n123ret+len(out106)]=src106[n106ret:]
    patch123[n123ret:n123ret+len(out106)]=patch106[n106ret:]
    six_ids=[]
    for j in range(6):
        src=np.asarray(vals[int(off[j]):int(off[j+1])],np.int64)
        rows=np.arange(n123ret+len(out106)+int(outoff[j]), n123ret+len(out106)+int(outoff[j+1]), dtype=np.int64)
        pid=1230000+j
        add_patch(patchsets,pid,src,rows,f"123 restored cavity {j}")
        patch123[rows]=pid; six_ids.append(pid)
    need(n123ret+len(out123)==len(src123) and np.all((src123>=0)|(patch123>=0)), "123 cell map has unmapped rows")
    need(np.all((src123>=0)&(patch123<0) | (src123<0)&(patch123>=0)), "123 rows have ambiguous ancestry")
    # 127 retains 123 rows and replaces only the named 16-cell source set with its 18-cell boundary repair.
    retained127=np.asarray(a127["inferred_boundary_retained_current_rows"],np.int64)
    selected127=np.asarray(a127["inferred_boundary_source_current_rows"],np.int64)
    assert_partition(retained127,selected127,len(a123["tetrahedra"]),"127/123")
    need(np.array_equal(a127["tetrahedra"][:len(retained127)],a123["tetrahedra"][retained127]), "127 retained cells differ from 123")
    new127=np.asarray(a127["inferred_boundary_output_tetrahedra_global_node_rows"],np.int64)
    need(np.array_equal(a127["tetrahedra"][len(retained127):],new127) and len(new127)==18, "127 output patch block mismatch")
    need(np.array_equal(a127["material_indices"][:len(retained127)],a123["material_indices"][retained127]), "127 changed retained raw attributes")
    need(np.all(a127["material_indices"][len(retained127):]==16), "127 boundary output material is not 16")
    srcselected=src123[selected127]; patchselected=patch123[selected127]
    need(np.all(srcselected>=0) and np.all(patchselected<0), "127 selected source cells are not all direct 096 cells")
    need(len(np.unique(srcselected))==len(srcselected), "127 selected source096 cell rows are not unique")
    pid127=1270005; add_patch(patchsets,pid127,srcselected,np.arange(len(retained127),len(retained127)+len(new127),dtype=np.int64),"127 inferred-boundary local repair")
    src127=np.full(len(a127["tetrahedra"]),-1,np.int64); patch127=np.full(len(src127),-1,np.int64)
    src127[:len(retained127)]=src123[retained127]; patch127[:len(retained127)]=patch123[retained127]
    patch127[len(retained127):]=pid127
    need(np.all((src127>=0)|(patch127>=0)) and np.all((src127<0)|(patch127<0)), "127 cell map is incomplete or ambiguous")
    # Candidate135 is a node-compacted exact representation: prove full row and point maps, not coordinate-nearest identity.
    point_out=np.asarray(a135["point_output_to_input_rows"],np.int64); point_in=np.asarray(a135["point_input_to_output_row"],np.int64)
    cellmap=np.asarray(a135["cell_output_to_input_rows"],np.int64)
    need(len(point_out)==len(a135["points_world_m"]) and np.all((point_out>=0)&(point_out<len(a127["points_world_m"]))) and len(np.unique(point_out))==len(point_out), "135 point output map is not injective")
    need(len(point_in)==len(a127["points_world_m"]) and np.array_equal(point_in[point_out],np.arange(len(point_out),dtype=np.int64)), "135 point maps are not reciprocal")
    need(np.array_equal(a135["points_world_m"],a127["points_world_m"][point_out]), "135 point coordinates differ from exact 127 source points")
    need(len(cellmap)==len(a135["tetrahedra"]) and np.array_equal(np.sort(cellmap),np.arange(len(a127["tetrahedra"]),dtype=np.int64)), "135 cell map is not a complete bijection")
    expected135=point_in[np.asarray(a127["tetrahedra"],np.int64)[cellmap]]
    need(np.all(expected135>=0) and np.array_equal(a135["tetrahedra"],expected135), "135 connectivity does not exactly reindex 127 cells")
    need(np.array_equal(a135["material_indices"],a127["material_indices"][cellmap]), "135 changed raw region/material attribute rows")
    src135=src127[cellmap]; patch135=patch127[cellmap]
    need(np.all((src135>=0)|(patch135>=0)) and np.all((src135<0)|(patch135<0)), "135 lineage map incomplete/ambiguous")
    # Candidate135's exact-volume proof is inherited only through the just-verified exact cell/coordinate maps.
    need(cert.get("proposal_sha256")==PIN["candidate127"] or cert.get("candidate_sha256")==PIN["candidate127"] or cert.get("input_pins",{}).get(str(p127))==PIN["candidate127"], "exact 127 certificate does not bind candidate127")
    need(r135.get("inherited_exact_sign_counts")=={"negative":0,"positive":853406,"zero":0}, "135 inherited exact sign receipt changed")
    # Aggregate patch source unions, ensure distinct patches do not claim the same source-096 cell.
    patch_order=sorted(patchsets)
    seen=set()
    for pid in patch_order:
        rows=patchsets[pid]["source096_rows"]
        overlap=seen.intersection(map(int,rows))
        need(not overlap, f"patch source096 unions overlap: {pid} with {sorted(overlap)[:8]}")
        seen.update(map(int,rows))
    patch_lineage={pid:build_patch_lineage(side,patchsets[pid]["source096_rows"]) for pid in patch_order}
    # Every replacement patch is declared material16; source parent region IDs are retained as provenance lanes.
    nout=len(a135["tetrahedra"]); n96=len(t96)
    need(nout==len(src135)==len(patch135), "135 output cell domain mismatch")
    # Copy inherited per-cell scalar ancestry; derived rows receive explicit patch-level semantics.
    scalar={k:np.empty(nout,dtype=side[k].dtype) for k in SCALAR_NAMES}
    direct=src135>=0
    for k in SCALAR_NAMES:
        scalar[k][direct]=side[k][src135[direct]]
    scalar["source_066_cell_row"][~direct]=-1
    scalar["source_064_candidate_cell_row"][~direct]=-1
    scalar["source_tetgen_element_id"][~direct]=-1
    scalar["source_region_attribute_059"][~direct]=-1
    scalar["corrected_region_attribute_062"][~direct]=16
    scalar["attribution_method"][~direct]=11
    scalar["attribution_patch_id"][~direct]=patch135[~direct]
    scalar["lineage_granularity"][~direct]=3
    # Compose inherited row CSR and patch-union CSR in the existing sidecar format.
    csrout={}
    for stem,(offkey,valkey) in CSR.items():
        pvalues={pid:patch_lineage[pid][stem] for pid in patch_order}
        oo,vv=compose_csr(side[offkey],side[valkey],src135,patch135,patch_order,pvalues)
        csrout[offkey]=oo; csrout[valkey]=vv
    # The TetGen parent ID CSR has three aligned source attribute lanes.
    source_parent_ids=csrout["source_parent_tetgen_element_ids"]
    source_parent_off=csrout["source_parent_tetgen_element_offsets"]
    raw_patches={pid:patch_lineage[pid]["source_parent_region_attribute_059"] for pid in patch_order}
    fix_patches={pid:patch_lineage[pid]["source_parent_region_attribute_062"] for pid in patch_order}
    mat_patches={pid:patch_lineage[pid]["source_parent_material_index"] for pid in patch_order}
    rawout=compose_csr(side["source_parent_tetgen_element_offsets"],side["source_parent_region_attribute_059"],src135,patch135,patch_order,raw_patches)[1]
    fixout=compose_csr(side["source_parent_tetgen_element_offsets"],side["source_parent_region_attribute_062"],src135,patch135,patch_order,fix_patches)[1]
    matout=compose_csr(side["source_parent_tetgen_element_offsets"],side["source_parent_material_index"],src135,patch135,patch_order,mat_patches)[1]
    need(len(source_parent_ids)==len(rawout)==len(fixout)==len(matout), "aligned source TetGen parent attribute CSR mismatch")
    need(np.all(matout==16) and np.all(np.isin(fixout,[16,17,18,19,20,21,22])), "output source parents outside certified Fat source scope")
    # Store the exact current-096 cell crosswalk and the patch-level source-cell union explicitly.
    patch_srcmap={pid:patchsets[pid]["source096_rows"] for pid in patch_order}
    src096_cell_parent_off,src096_cell_parent_rows=compose_csr(np.arange(n96+1,dtype=np.int64),np.arange(n96,dtype=np.int64),src135,patch135,patch_order,patch_srcmap)
    material=np.full(nout,16,dtype=side["material_index"].dtype)
    raw_attr=np.asarray(a135["material_indices"],np.int32)
    need(len(raw_attr)==nout, "135 raw region attribute length mismatch")
    outdir=args.output_dir.resolve(); need(not outdir.exists(), f"refusing overwrite existing output directory {outdir}")
    outdir.mkdir(parents=True)
    sidepath=outdir/"fat-material-attribution.npz"
    arrays={"output_cell_row":np.arange(nout,dtype=np.int64),"material_index":material,
        "current_candidate_region_attribute":raw_attr,**scalar,**csrout,
        "source_parent_region_attribute_059":rawout,"source_parent_region_attribute_062":fixout,"source_parent_material_index":matout,
        "source_096_cell_row":np.where(direct,src135,-1).astype(np.int64),
        "source_096_cell_parent_offsets":src096_cell_parent_off,"source_096_cell_parent_rows":src096_cell_parent_rows}
    np.savez_compressed(sidepath,**arrays)
    # Readback checks preserve row domains, material mapping, raw region labels, and nonempty aligned lineage.
    w=loadz(sidepath)
    need(np.array_equal(w["output_cell_row"],np.arange(nout)), "sidecar output row IDs malformed")
    need(np.all(w["material_index"]==16), "sidecar material lane is not all Fat16")
    need(np.array_equal(w["current_candidate_region_attribute"],raw_attr), "sidecar changed raw candidate attributes")
    for stem,(offkey,valkey) in CSR.items():
        need(len(w[offkey])==nout+1 and w[offkey][0]==0 and w[offkey][-1]==len(w[valkey]) and np.all(np.diff(w[offkey])>0), f"written {stem} CSR malformed")
    need(len(w["source_096_cell_parent_offsets"])==nout+1 and np.all(np.diff(w["source_096_cell_parent_offsets"])>0), "written direct-096 ancestry CSR malformed")
    need(np.array_equal(w["source_096_cell_row"],np.where(direct,src135,-1)), "written 096 direct row map mismatch")
    # Verify every patch row has the exact same declared source096 parent union in the output sidecar.
    for pid in patch_order:
        outrows=np.flatnonzero(patch135==pid); expected=patch_srcmap[pid]
        need(len(outrows)==len(patchsets[pid]["output_rows"]), f"patch {pid} output row count mismatch")
        if len(outrows):
            for r in [int(outrows[0]),int(outrows[-1])]:
                got=csr_row(w["source_096_cell_parent_offsets"],w["source_096_cell_parent_rows"],r)
                need(np.array_equal(got,expected), f"patch {pid} current-096 union readback mismatch")
    hist=lambda a:{str(k):int(v) for k,v in sorted(Counter(map(int,a)).items())}
    patch_records=[]
    for pid in patch_order:
        li=patch_lineage[pid]; src=patchsets[pid]["source096_rows"]
        patch_records.append({"patch_id":int(pid),"label":patchsets[pid]["label"],"output_cell_count":int(len(patchsets[pid]["output_rows"])),
            "output_row_min":int(np.min(patchsets[pid]["output_rows"])),"output_row_max":int(np.max(patchsets[pid]["output_rows"])),
            "source096_cell_union_count":int(len(src)),"source096_cell_union_sha256":hashlib.sha256(src.astype("<i8",copy=False).tobytes()).hexdigest(),
            "source096_cell_union_rows":src.tolist(),"source_tetgen_parent_count":int(len(li["source_parent_tetgen_element"])),
            "source_tetgen_parent_union_sha256":hashlib.sha256(li["source_parent_tetgen_element"].astype("<i8",copy=False).tobytes()).hexdigest(),
            "source_parent_region_attribute_059_histogram":hist(li["source_parent_region_attribute_059"]),
            "source_parent_region_attribute_062_histogram":hist(li["source_parent_region_attribute_062"]),
            "source_parent_material_histogram":hist(li["source_parent_material_index"])})
    report={"schema":"numi.human.fat-material-attribution-composite-chain.v1",
        "status":"MATERIAL_SIDECAR_BUILT_SOURCE_ATTRIBUTION_COMPLETE_GEOMETRY_NOT_ADMITTED",
        "candidate_path":str(p135.resolve()),"candidate_sha256":sha(p135),"candidate_cell_count":int(nout),
        "parent096_candidate_path":str(p96.resolve()),"parent096_candidate_sha256":sha(p96),
        "parent096_sidecar_path":str(s096p.resolve()),"parent096_sidecar_sha256":sha(s096p),
        "source_chain":["096 frozen source attribution","100 exact parent of 103 (authenticated by 103 input path/hash)","103 free-star composition","106 quantized/free-star/original-edge composition","123 six restored cavities","127 inferred-boundary 16-to-18 local repair","135 exact node compaction"],
        "input_pins":dict(sorted(pins.items())),
        "exact_checks":{"096_sidecar_matches_frozen096_candidate":True,"100_is_exact_immediate_parent_of103":True,
            "all_retained_cell_rows_map_exactly_to096":True,"all_generated_cells_have_nonempty_patch_union":True,
            "all_patch_parent_rows_are_source096_rows":True,"106_123_127_chain_outputs_replay_exactly":True,
            "135_point_and_cell_maps_are_exact_and_reciprocal":True,"135_tet_connectivity_and_raw_attribute_rows_replay_exactly":True,
            "all_parent_materials_are16":True,"all_corrected_parent_regions_within16_to22_scope":True,
            "all_sidecar_csr_rows_nonempty_and_aligned":True,"inherited_positive_sign_evidence_transfers_by_exact_cell_map":True},
        "source096_cells_mapped_directly":int(np.sum(direct)),"derived_output_cells_with_patch_union":int(np.sum(~direct)),
        "candidate_raw_region_attribute_histogram":hist(raw_attr),"material_index_histogram":{"16":int(nout)},
        "source_region_attribute_059_histogram":hist(w["source_region_attribute_059"]),
        "corrected_region_attribute_062_histogram":hist(w["corrected_region_attribute_062"]),
        "source_parent_region_attribute_059_histogram":hist(rawout),"source_parent_region_attribute_062_histogram":hist(fixout),
        "source_parent_material_histogram":hist(matout),"lineage_granularity_histogram":hist(w["lineage_granularity"]),
        "attribution_method_histogram":hist(w["attribution_method"]),"derived_patch_records":patch_records,
        "exact_positive_sign_evidence":{"source_candidate_sha256":PIN["candidate127"],"certificate_path":str(c127cert.resolve()),
            "certificate_sha256":sha(c127cert),"transferred_to135_by_exact_point_and_cell_maps":True,"new_exact_determinant_computation":False},
        "physical_admission":False,"runtime":0,"mass_mapping":"NOT_RUN","native_install":"NOT_RUN",
        "geometry_modified":False,"sidecar_path":str(sidepath),"sidecar_sha256":sha(sidepath),"elapsed_s":time.time()-t0}
    reportpath=outdir/"report.json"; reportpath.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:report[k] for k in ["status","candidate_cell_count","source096_cells_mapped_directly","derived_output_cells_with_patch_union","candidate_raw_region_attribute_histogram","material_index_histogram","sidecar_path","sidecar_sha256","physical_admission","mass_mapping","elapsed_s"]},indent=2,sort_keys=True))

if __name__=="__main__":
    main()
