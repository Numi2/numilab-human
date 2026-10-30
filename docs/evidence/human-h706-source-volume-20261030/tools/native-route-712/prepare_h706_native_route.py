#!/usr/bin/env python3
"""Hash-gated, non-executing launch-plan builder for H706 native Human volume."""
from __future__ import annotations
import argparse, hashlib, json, os, re, shlex, sys
from pathlib import Path
from typing import Any

P = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
BUILD = P / "operational-spatial-native-install-721/native-build-003"
SOURCE330 = P / "native-physical-parent-mass-330/native-source-identity-control-003"
BRIDGE = P / "operational-child-detail-bridge-691"
SOURCE_SNAPSHOT = P / "full-reference-native-capacity-705/cli-mass-mapping-task-002"
OVERLAY = P / "operational-spatial-native-install-721/overlay.json"
MANIFEST = P / "operational-spatial-native-install-721/source-manifest.json"
BINARY = BUILD / "numi-human-native"
PINNED_NATIVE = {
    str(BINARY): "f76e5373e05c1e0f3730bdc364b3f2bd946485b499ea1e9d065c2a550db201ef",
    str(BUILD / "compile-result.json"): "5eb9596f78bb9b0050877fb1b3a276ac1b35e4c373dd080c4d9f42fe93b6e2ac",
    str(BUILD / "link-result.json"): "147b7b7669173484d5f3021dc23b660e0122c4c7d16e6d34af1217bff02fd90f",
    str(OVERLAY): "477c4f4ecd1dbc5f2c0730400ef5fb8b528b6399137a855a07b76de6cff2f2ae",
    str(MANIFEST): "bce291c465f2b3e8def582c8926fe8321e785678968981c312f3c2ba95a338e2",
    str(SOURCE_SNAPSHOT / "source-snapshot/apps/NumiHumanPhysicalSkinAuthoringDetailBridgeCSV.hpp"):
        "2b74fdeddf38f17c34f995dc26d60507fa18f0be04db9af912d02e67e6aacafb",
    str(BRIDGE / "source-manifest-002.json"):
        "08c035049494d005fb1b511c659fb53c06d10e904ef38f485d59cb18858574a7",
}
PARENT = SOURCE330 / "compiled-parent-cell-mass.csv"
# The compiler receipt records the eight source-snapshot headers, overlay, and
# source manifest. The six external VFS contents are pinned independently here
# and bound directly by the recorded compile receipt where available.
VFS_CONTENT_PINS = {
    str(SOURCE_SNAPSHOT / "source-snapshot/apps/NumiHumanPhysicalSkinChildTetGen.hpp"): "dafb99dec977c2116ded1968ec0ae3ef493331109da6be6b19eb0f7148ea9e0c",
    str(SOURCE_SNAPSHOT / "source-snapshot/apps/NumiHumanUnpairedSkinFatVolume.hpp"): "2e3fbec0d1d6a737205d56868a801db40ab6a6ac1538eaef7c17e3f6213fc4fd",
    str(SOURCE_SNAPSHOT / "source-snapshot/apps/NumiHumanPhysicalSkinChildRegisteredFacetMaps.hpp"): "291f9679f30676e54a6ddde0283ecdcfb1b70403103d75651ff64f38b761070b",
    str(SOURCE_SNAPSHOT / "source-snapshot/apps/NumiHumanPhysicalSkinChildCapacity.hpp"): "9f099b6710224ad0eda48b31ce9e04c83a27142d171a7e0f38a3901d8d7932a6",
    str(SOURCE_SNAPSHOT / "source-snapshot/apps/NumiHumanPhysicalSkinChildRegisteredVolumeCSV.hpp"): "ed14fdab4cbfef3c8ea242273bd2548b9607feb7a8d3e7c1fa1cfaba799ed072",
    str(SOURCE_SNAPSHOT / "source-snapshot/apps/NumiHumanPhysicalSkinAuthoringRoleConsumer.hpp"): "6737dc0008aaae33ce09af58271d753073fd315b6ab267d5207a52ca2848a25d",
    str(SOURCE_SNAPSHOT / "source-snapshot/apps/NumiHumanPhysicalSkinAuthoringDetailBridgeCSV.hpp"): "2b74fdeddf38f17c34f995dc26d60507fa18f0be04db9af912d02e67e6aacafb",
    str(SOURCE_SNAPSHOT / "source-snapshot/apps/NumiHumanPhysicalSkinChildMassMapping.hpp"): "66c0e8eda2ea77a101286451045c9907194beeaef95411c5496789bf088cc848",
    str(P / "operational-spatial-native-install-721/source/apps/NumiHumanPhysicalSkinChildRegisteredVolumeIntegration.hpp"): "96341969bd6c84327c6bb546b0dcebe80726bf0dd165ccea2ad3c4c922f62228",
    str(P / "operational-spatial-reference-685/publication-baseline-dependencies/NumiHumanConservativeSpatialMassTransport.hpp"): "6873bdaa5722287a0542ed52b1a828279790ce2bd875f2c6156e98840909c3d7",
    str(P / "operational-spatial-native-install-721/source/apps/numilab_human_myosim_visual_probe.mm"): "e93faa5aea2fb2e1149ac13e06e71662e48b0130002635dd6995fd3a589fe63c",
    str(P / "operational-spatial-reference-685/publication-baseline-dependencies/NumiHumanPhysicalSkinChildRegisteredVolumeImport.hpp"): "04e54b1c7b1fbea383b61b5d5df5dfbf9dd0fe0334097bb2c0a825eb5b70a4d4",
    str(P / "canonical-attachment-native-integration-688/publication-source/apps/NumiHumanCanonicalDeepAttachmentRemap.hpp"): "972dc2e1ed48d5a555da69fe0a145c17293b72d0f7325925e8b6c73ff3416b6e",
    str(P / "operational-child-detail-bridge-691/publication-source/apps/NumiHumanPhysicalSkinAuthoringDetailBridgeCSVCheck.cpp"): "a18d0095711295a0a33643d1223dabee152e43ee486d233bf8d82aede7856a33",
}
RUNTIME_ASSET_PINS = {
    "/Users/numi/Work/skin-fat-native-integration-209/build-native/lib/libmetalrobo.dylib": "b4b470ec60f80e7edc8f6c4b7c1eb5cc0296beeba8601332a05a1054d7db5e4e",
    "/Users/numi/Work/skin-fat-native-integration-209/build-native/shaders/MetalRobo.metallib": "b52b5c1c5736da382682e4412aa9b458603066c0c5728a8869d02627aef97672",
    "/Users/numi/Work/skin-fat-native-integration-209/build-native/matter/shaders/HumanRespiration.metallib": "9b96bb0eccab8d71cc2655cfff07622330267e95630a19776798beda36bbc3ff",
    "/Users/numi/Work/skin-fat-native-integration-209/build-native/matter/shaders/NumiMatter.metallib": "76819990f8b17e27621b2e463ff953baa04f05f1bd0aaa94d27f51fbbac0979b",
    "/Users/numi/Work/skin-fat-native-integration-209/build-native/matter/shaders/NumiMatterPhysicalStateDigest.metallib": "28665e39f1595d78fb4fd3c4c22d57ce5aee0f57f956d8211f0b8d2aae5dc16c",
    "/Users/numi/Work/skin-fat-native-integration-209/build-native/shaders/MetalRoboHyperPolicy.metallib": "8d5676406a80a71ea50828fad47e78083892c1bc996299c59253d2a398cdd6d9",
    "/Users/numi/Work/skin-fat-native-integration-209/build-native/shaders/NumiNeuron.metallib": "965289118c76f59072d46535b9fe96a5eb8e56a50596705609cbff84d50617f1",
}
PINNED_PARENT = {
    str(PARENT): "c7823c42d64a2154a34c87cc0d4d202f0d5256f79217e74f5f62bcf1d5b31679",
    str(PARENT) + ".labels.csv": "99d643f71a735ee49223cdcd7b863d2d0978b8cc490a138bad4a63012438a457",
    str(PARENT) + ".nodes.csv": "7044051353d173b94143a4027ba7dcc468b221512b3f41d5819d27025b36d044",
    str(PARENT) + ".source.csv": "8e2424901318f25caca1540aeca412ca779fc014f68f49fe0cb6c17fb5aa0c1f",
    str(PARENT) + ".tets.csv": "82e20067454748a9eca91bc6f06ab343677c051a139797ae051c64022bd87992",
    str(PARENT) + ".weights.csv": "98205ae7de44401544b7969184788f5c2ee39cd4404cd34b8037377e7ae63a76",
    str(SOURCE330 / "registered-skin-poses-100.json"):
        "0c6e10d411604cb317eb5a9da234da568d683e6cc0276ac184f80fd29067b80a",
}
MAX_MASS_TRANSPORT_M = "0.019353"
MAX_NATIVE_PARENT_REGISTRATION_M = "0.0004"
PREFIX = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
REQUIRED_ENV = {
    "PARENT_PACKET_CSV","PARENT_PACKET_SHA256","PARENT_LABELS_CSV","PARENT_LABELS_SHA256",
    "PARENT_NODES_SHA256","PARENT_SOURCE_SHA256","PARENT_TETS_SHA256",
    "NODE_REGISTRATION_CSV","NODE_REGISTRATION_SHA256","DONOR_FRAMES_CSV","DONOR_FRAMES_SHA256",
    "OUTER_FACE_ANCESTRY_CSV","OUTER_FACE_ANCESTRY_SHA256",
    "D37_INNER_FACE_COUNT","D37_DEEP_FACE_COUNT","MAX_PARENT_REGISTRATION_DISTANCE_M",
    "MAXIMUM_MASS_TRANSPORT_DISTANCE_M","COORDINATE_FRAME_SHA256",
    "DEEP_ATTACHMENT_REQUESTS_CSV","DEEP_ATTACHMENT_REQUESTS_SHA256",
    "TETGEN_NODE","TETGEN_NODE_SHA256","TETGEN_ELEMENTS","TETGEN_ELEMENTS_SHA256",
    "CHILD_LABELS","CHILD_LABELS_SHA256","COMBINED_NODE_MAP_CSV","COMBINED_NODE_MAP_SHA256",
    "FACET_ANCESTRY_CSV","FACET_ANCESTRY_SHA256",
    "OPERATIONAL_SOURCE_NODE","OPERATIONAL_SOURCE_NODE_SHA256",
    "OPERATIONAL_SOURCE_ELEMENTS","OPERATIONAL_SOURCE_ELEMENTS_SHA256",
    "OPERATIONAL_SOURCE_LABELS","OPERATIONAL_SOURCE_LABELS_SHA256",
    "OPERATIONAL_SOURCE_VOLUME_ASSET","OPERATIONAL_SOURCE_VOLUME_SHA256",
    "OPERATIONAL_SOURCE_PRODUCER_RECEIPT","OPERATIONAL_SOURCE_PRODUCER_RECEIPT_SHA256",
    "OPERATIONAL_SOURCE_SELF_AUDIT_RECEIPT","OPERATIONAL_SOURCE_SELF_AUDIT_RECEIPT_SHA256",
    "OPERATIONAL_CANDIDATE_VOLUME_ASSET","OPERATIONAL_CANDIDATE_VOLUME_SHA256",
    "OPERATIONAL_CANDIDATE_PRODUCER_RECEIPT","OPERATIONAL_CANDIDATE_PRODUCER_RECEIPT_SHA256",
    "OPERATIONAL_CANDIDATE_SELF_AUDIT_RECEIPT","OPERATIONAL_CANDIDATE_SELF_AUDIT_RECEIPT_SHA256",
    "DETAIL_SOURCE_ASSET","DETAIL_SOURCE_ASSET_SHA256","DETAIL_SOURCE_DOMAIN",
    "DETAIL_SOURCE_VERTEX_COUNT","DETAIL_SOURCE_FACE_COUNT",
    "DETAIL_NODES_CSV","DETAIL_NODES_SHA256","DETAIL_FACES_CSV","DETAIL_FACES_SHA256",
    "DETAIL_FULL86_CSV","DETAIL_FULL86_SHA256","DETAIL_FULL86_OWNERS_CSV","DETAIL_FULL86_OWNERS_SHA256",
    "DETAIL_TO_BOUNDARY_CSV","DETAIL_TO_BOUNDARY_SHA256","DETAIL_TO_BOUNDARY_CANONICAL_EXPORT_CSV",
    "DETAIL_TO_BOUNDARY_HISTORICAL_CSV","DETAIL_TO_BOUNDARY_HISTORICAL_SHA256",
    "COARSE_BOUNDARY_FACES_CSV","COARSE_BOUNDARY_FACES_SHA256",
}
for role in ("O","I","D"):
    REQUIRED_ENV.update({
        f"SURFACE_{role}_NODES_CSV",f"SURFACE_{role}_NODES_SHA256",
        f"SURFACE_{role}_FACES_CSV",f"SURFACE_{role}_FACES_SHA256",
        f"SURFACE_{role}_STABLE_IDENTIFIER",
        f"ROLE_{role}_SOURCE_DOMAIN",f"ROLE_{role}_SOURCE_ASSET",
        f"ROLE_{role}_SOURCE_ASSET_SHA256",f"ROLE_{role}_SOURCE_VERTEX_COUNT",
        f"ROLE_{role}_SOURCE_FACE_COUNT",
        f"ROLE_{role}_FULL86_CSV",f"ROLE_{role}_FULL86_SHA256",
        f"ROLE_{role}_CONTACT_THICKNESS_CSV",f"ROLE_{role}_CONTACT_THICKNESS_SHA256",
    })
# These admission checks are semantic requirements; each manifest maps them to
# concrete paths in the current candidate-specific gate receipt.
ROLE_SELF_COVERAGE_CLAIM_NAMES = {
    f"{role}_{suffix}"
    for role in ("O", "I", "D")
    for suffix in ("full_self_pairs", "full_self_selected_faces",
                   "full_self_complete_faces", "candidate_faces",
                   "full_self_status")
}
REQUIRED_GEOMETRY_CLAIMS = {
    "physical_admission", "positive_f32_cells", "closed_skin", "closed_fat",
    "shared_I_exact_opposed", "O_I_exact_pairs", "I_D_exact_pairs",
    "skin_fat_exact_pairs", "native_f32_bone_pairs", "pinned_f64_bone_pairs",
    "all185_operand_coverage", "all149_protected_routes", "existing_thickness_bounds",
} | ROLE_SELF_COVERAGE_CLAIM_NAMES
REQUIRED_MASS_CLAIMS = {
    "mass_mapping_pass", "mass_conservation_pass", "all_material_density_bounds",
    "regional_capacity_pass", "transport_distance_bounds",
}

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def dump(path: Path, obj: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite saved output: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n")

def read(path: Path) -> Any:
    return json.loads(path.read_text())

def checked_hash(path: str, expected: str, label: str) -> str:
    p=Path(path)
    if not p.is_file(): raise ValueError(f"{label}: missing file {p}")
    actual=sha256(p)
    if actual != expected: raise ValueError(f"{label}: SHA-256 mismatch for {p}: {actual}")
    return actual

def verify_vfs_content_pins(overlay: dict, compile_info: dict, compile_receipt: Path,
                           pins: dict[str,str]) -> dict[str,dict]:
    roots=overlay.get("roots")
    if not isinstance(roots,list) or len(roots)!=16:
        raise ValueError("native721 VFS overlay must contain exactly 16 file mappings")
    mapped={}
    names=set()
    for root in roots:
        if not isinstance(root,dict) or root.get("type")!="file":
            raise ValueError("native721 VFS overlay contains a non-file root")
        name=root.get("name"); source=root.get("external-contents")
        if not isinstance(name,str) or not isinstance(source,str) or name in names:
            raise ValueError("native721 VFS overlay root is malformed or duplicates a VFS name")
        names.add(name)
        mapped.setdefault(source,[]).append(name)
    if set(mapped)!=set(pins) or len(mapped)!=14:
        raise ValueError("native721 VFS overlay must resolve to the 14 pinned unique external contents")
    compiled=compile_info.get("input_sha256",{})
    if not isinstance(compiled,dict):
        raise ValueError("native721 compiler receipt lacks its input hash map")
    compiled_ns=compile_receipt.stat().st_mtime_ns
    report={}
    for source,expected in pins.items():
        actual=checked_hash(source,expected,"native721 VFS content")
        if source in compiled and compiled[source]!=actual:
            raise ValueError(f"native721 compiler receipt hash mismatch for VFS source: {source}")
        if source not in compiled and Path(source).stat().st_mtime_ns>compiled_ns:
            raise ValueError(f"external VFS source changed after native721 compile receipt: {source}")
        report[source]={"vfs_names":sorted(mapped[source]),"sha256":actual,
                        "compiler_receipt_hash":compiled.get(source),
                        "compile_time_guard":"receipt_hash" if source in compiled else "source_mtime_precedes_compile_receipt"}
    return report

def verify_pinned_static() -> dict[str,str]:
    result={}
    all_pins={**PINNED_NATIVE,**PINNED_PARENT,**VFS_CONTENT_PINS,**RUNTIME_ASSET_PINS}
    for path,expected in all_pins.items():
        result[path]=checked_hash(path,expected,"pinned current native/source330 input")
    compile_path=BUILD/"compile-result.json"
    link_path=BUILD/"link-result.json"
    compile_info=read(compile_path)
    link_info=read(link_path)
    if compile_info.get("exit_code")!=0 or link_info.get("exit_code")!=0:
        raise ValueError("native721 build receipts do not report successful compile and link")
    source_manifest=read(MANIFEST)
    if source_manifest.get("schema")!="numi.full-reference-native-capacity-source-manifest.v2":
        raise ValueError("native721 source manifest is not the pinned full-reference capacity increment")
    if source_manifest.get("physical_admission") is not False or source_manifest.get("runtime")!=0:
        raise ValueError("native721 source manifest must remain source-only")
    if compile_info.get("runtime")!=0 or link_info.get("runtime")!=0:
        raise ValueError("native721 receipt unexpectedly reports Runtime execution")
    if compile_info.get("physical_admission") is not False or link_info.get("physical_admission") is not False:
        raise ValueError("native721 build receipts unexpectedly claim physical admission")
    source_dir=P/"operational-spatial-native-install-721/source/apps"
    source_files=source_manifest.get("files",{})
    required_sources={"NumiHumanPhysicalSkinChildRegisteredVolumeIntegration.hpp",
                      "NumiHumanPhysicalSkinChildRegisteredVolumeIntegrationCheck.cpp",
                      "numilab_human_myosim_visual_probe.mm"}
    if not isinstance(source_files,dict) or not required_sources.issubset(source_files):
        raise ValueError("native721 source manifest lacks the runtime installer source and tests")
    for name,digest in source_files.items():
        checked_hash(str(source_dir/name),digest,"native721 source manifest")
    if link_info.get("binary_sha256")!=PINNED_NATIVE[str(BINARY)]:
        raise ValueError("native721 link receipt does not bind the pinned binary")
    vfs_report=verify_vfs_content_pins(read(OVERLAY),compile_info,compile_path,VFS_CONTENT_PINS)
    link_ns=link_path.stat().st_mtime_ns
    for path in RUNTIME_ASSET_PINS:
        if Path(path).stat().st_mtime_ns>link_ns:
            raise ValueError(f"runtime dylib/metallib is newer than the native721 link receipt: {path}")
    for path,expected in compile_info.get("input_sha256",{}).items():
        checked_hash(path,expected,"native721 compile input")
    result["__vfs_external_content_count__"]=str(len(vfs_report))
    result["__vfs_external_content__"]=vfs_report
    result["__compiled_probe_source_sha256__"]=VFS_CONTENT_PINS[str(P/"operational-spatial-native-install-721/source/apps/numilab_human_myosim_visual_probe.mm")]
    return result

def get_path(obj: Any, dotted: str) -> Any:
    cur=obj
    for key in dotted.split("."):
        if not isinstance(cur,dict) or key not in cur:
            raise ValueError(f"receipt lacks claim path {dotted}")
        cur=cur[key]
    return cur

def validate_complete_role_self_coverage(claims: dict) -> None:
    """Require full current candidate-face coverage for O, I, and D self audits."""
    if not isinstance(claims,dict) or not ROLE_SELF_COVERAGE_CLAIM_NAMES.issubset(claims):
        raise ValueError("geometry admission requires a complete exact-gate report for O/I/D")
    for role in ("O","I","D"):
        expected_paths = {
            f"{role}_full_self_pairs": f"full_current_self_results.{role}.count",
            f"{role}_full_self_selected_faces": f"stages.full_current_{role}_self.selected_faces",
            f"{role}_full_self_complete_faces": f"stages.full_current_{role}_self.complete_faces",
            f"{role}_candidate_faces": f"candidate_readback.{role}.candidate_faces",
            f"{role}_full_self_status": f"stages.full_current_{role}_self.status",
        }
        for name, expected_path in expected_paths.items():
            claim=claims[name]
            if not isinstance(claim,dict) or set(("path","equals"))-set(claim):
                raise ValueError(f"geometry admission: malformed complete self-coverage claim {name}")
            if claim["path"] != expected_path:
                raise ValueError(f"geometry admission: {name} must bind {expected_path}")
        pair_claim=claims[f"{role}_full_self_pairs"]
        if type(pair_claim["equals"]) is not int or pair_claim["equals"] != 0:
            raise ValueError(f"geometry admission: full {role} self-pair count must be integer zero")
        face_claims=[
            claims[f"{role}_full_self_selected_faces"]["equals"],
            claims[f"{role}_full_self_complete_faces"]["equals"],
            claims[f"{role}_candidate_faces"]["equals"],
        ]
        if any(type(count) is not int or count <= 0 for count in face_claims):
            raise ValueError(f"geometry admission: {role} full self-coverage face counts must be positive integers")
        if len(set(face_claims)) != 1:
            raise ValueError(f"geometry admission: {role} full self audit does not cover every current candidate face")
        if claims[f"{role}_full_self_status"]["equals"] != "PASS":
            raise ValueError(f"geometry admission: full {role} self audit status must be PASS")

def require_no_failure_markers(stdout: str, stderr: str, markers: tuple[str,...],
                               stage: str) -> None:
    combined=stdout+"\n"+stderr
    found=[marker for marker in markers if marker in combined]
    if found:
        raise ValueError(f"{stage} log contains failure marker(s): {found}")

def check_claim_receipt(spec: dict, required: set[str], label: str) -> tuple[dict,str]:
    path=spec.get("path"); expected=spec.get("sha256")
    if not isinstance(path,str) or not isinstance(expected,str):
        raise ValueError(f"{label}: receipt path and SHA-256 are required")
    checked_hash(path,expected,label)
    receipt=read(Path(path))
    claims=spec.get("claims",{})
    missing=required-set(claims)
    if missing: raise ValueError(f"{label}: missing required claims {sorted(missing)}")
    if label=="geometry admission":
        validate_complete_role_self_coverage(claims)
    for name,claim in claims.items():
        if not isinstance(claim,dict) or set(("path","equals"))-set(claim):
            raise ValueError(f"{label}: malformed claim {name}")
        actual=get_path(receipt,claim["path"])
        wanted=claim["equals"]
        if actual != wanted:
            raise ValueError(f"{label}: claim {name} expected {wanted!r}, got {actual!r}")
        if name.endswith("_pairs") and actual != 0:
            raise ValueError(f"{label}: exact pair gate {name} is nonzero")
    if label=="geometry admission" and claims["physical_admission"]["equals"] is not True:
        raise ValueError("geometry receipt must explicitly attest physical_admission=true")
    return receipt,expected

def verify_input_pins(pins: dict[str,str]) -> dict[str,str]:
    if not isinstance(pins,dict) or not pins:
        raise ValueError("input_pins must map every current asset path to its SHA-256")
    checked={}
    for path,digest in pins.items():
        if not re.fullmatch(r"[0-9a-f]{64}",str(digest)):
            raise ValueError(f"invalid SHA-256 pin for {path}")
        checked[path]=checked_hash(path,digest,"current H706 input")
    return checked

def input_pin_diagnostics(pins: Any) -> list[str]:
    if not isinstance(pins,dict) or not pins:
        return ["input_pins must map every current asset path to its SHA-256"]
    errors=[]
    for path,digest in pins.items():
        if not isinstance(path,str) or not re.fullmatch(r"[0-9a-f]{64}",str(digest)):
            errors.append(f"current H706 input: invalid path or SHA-256 pin for {path}")
            continue
        try:
            checked_hash(path,digest,"current H706 input")
        except (ValueError,TypeError) as exc:
            errors.append(str(exc))
    return errors

def validate_environment(env: dict[str,str], pins: dict[str,str], detail_export_path: str) -> dict[str,str]:
    missing=REQUIRED_ENV-set(env)
    if missing: raise ValueError(f"native preflight environment lacks required fields {sorted(missing)}")
    if env["MAXIMUM_MASS_TRANSPORT_DISTANCE_M"] != MAX_MASS_TRANSPORT_M:
        raise ValueError("mass transport distance differs from the existing 0.019353 m bound")
    if env["MAX_PARENT_REGISTRATION_DISTANCE_M"] != MAX_NATIVE_PARENT_REGISTRATION_M:
        raise ValueError("native parent registration differs from the current 0.0004 m bound")
    if env["OPERATIONAL_CONFORMING_REFINEMENT_ACTUAL_PROOF_RECEIPT"] if "OPERATIONAL_CONFORMING_REFINEMENT_ACTUAL_PROOF_RECEIPT" in env else False:
        raise ValueError("H706 split operational route must not claim a stale 530 conforming-refinement proof")
    if env["OPERATIONAL_CONFORMING_REFINEMENT_SELF_AUDIT_RECEIPT"] if "OPERATIONAL_CONFORMING_REFINEMENT_SELF_AUDIT_RECEIPT" in env else False:
        raise ValueError("H706 split operational route must not claim a stale 530 conforming-refinement audit")
    # Parent mass source, labels, and frame are fixed to the current 330 source.
    for key,path in (("PARENT_PACKET_CSV",str(PARENT)),
                     ("PARENT_LABELS_CSV",str(PARENT)+".labels.csv")):
        if env[key] != path: raise ValueError(f"{key} must use the pinned source330 packet")
    for suffix,field in (("PARENT_PACKET_SHA256",str(PARENT)),
                         ("PARENT_LABELS_SHA256",str(PARENT)+".labels.csv"),
                         ("PARENT_NODES_SHA256",str(PARENT)+".nodes.csv"),
                         ("PARENT_SOURCE_SHA256",str(PARENT)+".source.csv"),
                         ("PARENT_TETS_SHA256",str(PARENT)+".tets.csv")):
        if env[suffix] != PINNED_PARENT[field]:
            raise ValueError(f"{suffix} differs from the immutable source330 pin")
    if env["COORDINATE_FRAME_SHA256"] != PINNED_PARENT[str(SOURCE330/"registered-skin-poses-100.json")]:
        raise ValueError("coordinate frame differs from the pinned source330 pose snapshot")
    path_hash_pairs={
        "PARENT_PACKET_CSV":"PARENT_PACKET_SHA256","PARENT_LABELS_CSV":"PARENT_LABELS_SHA256",
        "NODE_REGISTRATION_CSV":"NODE_REGISTRATION_SHA256","DONOR_FRAMES_CSV":"DONOR_FRAMES_SHA256",
        "OUTER_FACE_ANCESTRY_CSV":"OUTER_FACE_ANCESTRY_SHA256",
        "DEEP_ATTACHMENT_REQUESTS_CSV":"DEEP_ATTACHMENT_REQUESTS_SHA256",
        "TETGEN_NODE":"TETGEN_NODE_SHA256","TETGEN_ELEMENTS":"TETGEN_ELEMENTS_SHA256",
        "CHILD_LABELS":"CHILD_LABELS_SHA256","COMBINED_NODE_MAP_CSV":"COMBINED_NODE_MAP_SHA256",
        "FACET_ANCESTRY_CSV":"FACET_ANCESTRY_SHA256",
        "OPERATIONAL_SOURCE_NODE":"OPERATIONAL_SOURCE_NODE_SHA256",
        "OPERATIONAL_SOURCE_ELEMENTS":"OPERATIONAL_SOURCE_ELEMENTS_SHA256",
        "OPERATIONAL_SOURCE_LABELS":"OPERATIONAL_SOURCE_LABELS_SHA256",
        "OPERATIONAL_SOURCE_VOLUME_ASSET":"OPERATIONAL_SOURCE_VOLUME_SHA256",
        "OPERATIONAL_SOURCE_PRODUCER_RECEIPT":"OPERATIONAL_SOURCE_PRODUCER_RECEIPT_SHA256",
        "OPERATIONAL_SOURCE_SELF_AUDIT_RECEIPT":"OPERATIONAL_SOURCE_SELF_AUDIT_RECEIPT_SHA256",
        "OPERATIONAL_CANDIDATE_VOLUME_ASSET":"OPERATIONAL_CANDIDATE_VOLUME_SHA256",
        "OPERATIONAL_CANDIDATE_PRODUCER_RECEIPT":"OPERATIONAL_CANDIDATE_PRODUCER_RECEIPT_SHA256",
        "OPERATIONAL_CANDIDATE_SELF_AUDIT_RECEIPT":"OPERATIONAL_CANDIDATE_SELF_AUDIT_RECEIPT_SHA256",
        "DETAIL_SOURCE_ASSET":"DETAIL_SOURCE_ASSET_SHA256","DETAIL_NODES_CSV":"DETAIL_NODES_SHA256",
        "DETAIL_FACES_CSV":"DETAIL_FACES_SHA256","DETAIL_FULL86_CSV":"DETAIL_FULL86_SHA256",
        "DETAIL_FULL86_OWNERS_CSV":"DETAIL_FULL86_OWNERS_SHA256",
        "DETAIL_TO_BOUNDARY_CSV":"DETAIL_TO_BOUNDARY_SHA256",
        "DETAIL_TO_BOUNDARY_HISTORICAL_CSV":"DETAIL_TO_BOUNDARY_HISTORICAL_SHA256",
        "COARSE_BOUNDARY_FACES_CSV":"COARSE_BOUNDARY_FACES_SHA256",
    }
    for role in ("O","I","D"):
        path_hash_pairs[f"SURFACE_{role}_NODES_CSV"]=f"SURFACE_{role}_NODES_SHA256"
        path_hash_pairs[f"SURFACE_{role}_FACES_CSV"]=f"SURFACE_{role}_FACES_SHA256"
        path_hash_pairs[f"ROLE_{role}_SOURCE_ASSET"]=f"ROLE_{role}_SOURCE_ASSET_SHA256"
        path_hash_pairs[f"ROLE_{role}_FULL86_CSV"]=f"ROLE_{role}_FULL86_SHA256"
        path_hash_pairs[f"ROLE_{role}_CONTACT_THICKNESS_CSV"]=f"ROLE_{role}_CONTACT_THICKNESS_SHA256"
    for path_key,hash_key in path_hash_pairs.items():
        file_path=env[path_key]
        if file_path not in pins: raise ValueError(f"{path_key} is not SHA-pinned in input_pins")
        if pins[file_path] != env[hash_key]:
            raise ValueError(f"{path_key} path pin differs from {hash_key}")
    for role in ("O","I","D"):
        for key in (f"SURFACE_{role}_NODES_CSV",f"SURFACE_{role}_FACES_CSV",
                    f"ROLE_{role}_SOURCE_ASSET",f"ROLE_{role}_FULL86_CSV",
                    f"ROLE_{role}_CONTACT_THICKNESS_CSV"):
            path=env[key]
            if path not in pins: raise ValueError(f"{key} path is not SHA-pinned in input_pins")
            if pins[path] != env[key.replace("_CSV","_SHA256").replace("_ASSET","_ASSET_SHA256")]:
                raise ValueError(f"{key} does not match its companion SHA field")
    # Verify every registered-volume file and receipt, not just the top-level 5.
    for key,value in env.items():
        if key.endswith(("_CSV","_NODE","_ELEMENTS","_ASSET","_RECEIPT")):
            if key=="DETAIL_TO_BOUNDARY_CANONICAL_EXPORT_CSV":
                continue
            if value not in pins: raise ValueError(f"preflight path {key} lacks an exact input SHA pin")
    if not detail_export_path.startswith("/"):
        raise ValueError("detail export target must be an absolute path")
    return env.copy()

def validate_scene_output_slot(argv: list[str]) -> None:
    """Validate the probe's fourth free positional argument, not argv tail."""
    if argv.count("{RUN_OUTPUT}") != 1:
        raise ValueError("scene.argv must contain exactly one {RUN_OUTPUT} argument slot")
    if len(argv) < 4 or argv[3] != "{RUN_OUTPUT}":
        raise ValueError("scene.argv must place {RUN_OUTPUT} fourth, after rigid, muscle, and bone inputs")
    if any(not value.startswith("/") for value in argv[:3]):
        raise ValueError("the first three scene arguments must be absolute rigid, muscle, and bone payload paths")
    if "--resting-movie" in argv:
        i=argv.index("--resting-movie")
        if argv.count("--resting-movie")!=1 or i+1>=len(argv) or argv[i+1]!="{RUN_MOVIE}":
            raise ValueError("scene --resting-movie output must use the single {RUN_MOVIE} stage path")
        if argv.count("{RUN_MOVIE}")!=1:
            raise ValueError("scene argv must contain exactly one {RUN_MOVIE} output slot")
    elif "{RUN_MOVIE}" in argv:
        raise ValueError("{RUN_MOVIE} requires the --resting-movie option")

def expand_scene_argv(argv: list[str], stage_dir: Path) -> list[str]:
    movie=str(stage_dir/"native-viewer.mov")
    return [value.replace("{RUN_OUTPUT}",str(stage_dir)).replace("{RUN_MOVIE}",movie)
            for value in argv]

def expand_stage_environment(env: dict[str,str], stage_dir: Path) -> dict[str,str]:
    movie=str(stage_dir/"native-viewer.mov")
    return {key:(value.replace("{RUN_OUTPUT}",str(stage_dir)).replace("{RUN_MOVIE}",movie)
                 if isinstance(value,str) else value)
            for key,value in env.items()}

def argv_step_count(argv: list[str]) -> int:
    if argv.count("--muscle-step-count") != 1:
        raise ValueError("scene argv must contain exactly one --muscle-step-count")
    try:
        i=argv.index("--muscle-step-count")
        value=argv[i+1]
        if not re.fullmatch(r"[0-9]+",value):
            raise ValueError
        n=int(value)
        if n<=0: raise ValueError
        return n
    except (ValueError,IndexError):
        raise ValueError("scene argv must request one positive integer --muscle-step-count")

def assert_plan_matches_regenerated(plan: dict, expected_plan: dict) -> None:
    if plan != expected_plan:
        raise ValueError("saved launch plan differs from the manifest-derived route")

def verify_saved_plan_inputs(plan: dict) -> None:
    verify_pinned_static()
    verify_input_pins(plan.get("input_pins",{}))
    geometry=plan.get("geometry_admission",{})
    mass=plan.get("mass_admission",{})
    check_claim_receipt(geometry,REQUIRED_GEOMETRY_CLAIMS,"geometry admission")
    check_claim_receipt(mass,REQUIRED_MASS_CLAIMS,"mass admission")
    manifest_path=plan.get("manifest_path")
    if not isinstance(manifest_path,str):
        raise ValueError("plan lacks its original H706 manifest path")
    resolved_manifest=Path(manifest_path).resolve()
    manifest=read(resolved_manifest)
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if digest!=plan.get("manifest_sha256"):
        raise ValueError("H706 manifest changed after launch-plan generation")

    source_receipt=None
    receipt_path=plan.get("source_preflight_receipt_path")
    receipt_sha=plan.get("source_preflight_receipt_sha256")
    if receipt_path is not None or receipt_sha is not None:
        if not isinstance(receipt_path,str) or not isinstance(receipt_sha,str):
            raise ValueError("saved motion plan has an incomplete source-preflight receipt pin")
        checked_hash(receipt_path,receipt_sha,"source-only receipt")
        source_receipt=read(Path(receipt_path))
    expected=build_plan(manifest,resolved_manifest.parent,source_receipt,
                        require_fresh_outputs=False)
    expected["manifest_sha256"]=digest
    expected["manifest_path"]=str(resolved_manifest)
    if source_receipt is not None:
        expected["source_preflight_receipt_path"]=str(Path(receipt_path).resolve())
        expected["source_preflight_receipt_sha256"]=receipt_sha
    assert_plan_matches_regenerated(plan,expected)

def validate_physical_spatial_install(scene_env: dict[str,str]) -> None:
    install_key="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_OPERATIONAL_SPATIAL_INSTALL"
    split_key="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_OPERATIONAL_REFERENCE_SPLIT"
    spatial_key="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_OPERATIONAL_CONSERVATIVE_SPATIAL_REFERENCE"
    source_only_key="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_SOURCE_ONLY"
    preflight_key="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT"
    # Physical operational-reference installation is an explicit opt-in. It
    # keeps the source330 history fields and uses the verified conservative
    # spatial mass route; it is distinct from the zero-runtime source-only
    # preflight even though both carry the split/spatial proof flags.
    required={install_key:"1",split_key:"1",spatial_key:"1",
              source_only_key:"0",preflight_key:"1"}
    for key,expected in required.items():
        if scene_env.get(key)!=expected:
            raise ValueError(f"physical operational spatial install requires {key}={expected}")
    if scene_env.get("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_OPERATIONAL_CONFORMING_REFINEMENT_CHAIN")=="1":
        raise ValueError("physical operational spatial install excludes the conforming-refinement route")

def validate_full_scene_mode(argv: list[str], scene_env: dict[str,str]) -> None:
    forbidden={"--mechanics-only","--parent-mass-capture-only",
               "--skin-fat-parent-mass-capture-only","--identity-import",
               "--identity-import-only"}
    bad=sorted(forbidden.intersection(argv))
    if bad:
        raise ValueError(f"full Human motion route forbids preparation/capture modes: {bad}")
    if scene_env.get("NUMI_HUMAN_SKIN_FAT_PARENT_MASS_CAPTURE_ONLY")=="1":
        raise ValueError("full Human motion route forbids source330 parent-mass capture-only mode")
    if scene_env.get("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_SOURCE_ONLY")=="1":
        raise ValueError("source-only import is limited to the separate zero-runtime preflight stage")
    if scene_env.get("NUMI_HUMAN_RESTING_PASSIVE_FASCIA")=="0":
        raise ValueError("registered skin/fat motion requires the intact resting tissue world")

def build_stage_env(scene_env: dict[str,str], suffix_env: dict[str,str], *, source_only: bool,
                    canonical_export: bool, operational_split: bool,
                    operational_spatial_install: bool=False) -> dict[str,str]:
    result=dict(scene_env)
    result.update({PREFIX+k:str(v) for k,v in suffix_env.items()})
    result["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_INSTALL"]="1"
    result["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_OPERATIONAL_SPATIAL_INSTALL"]=(
        "1" if operational_spatial_install else "0")
    result["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_OPERATIONAL_REFERENCE_SPLIT"]="1" if operational_split else "0"
    result["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_OPERATIONAL_CONSERVATIVE_SPATIAL_REFERENCE"]=(
        "1" if operational_split else "0")
    result["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT"]="1"
    result["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_SOURCE_ONLY"]="1" if source_only else "0"
    result["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_DETAIL_BRIDGE"]="1"
    result["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_DETAIL_BRIDGE_CANONICAL_FACE_EXPORT"]="1" if canonical_export else "0"
    result["NUMI_HUMAN_RESTING_PASSIVE_FASCIA"]="1"
    result["NUMI_HUMAN_RESTING_SKIN_FAT_FEM"]="1"
    result["NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM"]="1"
    result["NUMI_HUMAN_SKIN_FAT_PARENT_MASS_CAPTURE_ONLY"]="0"
    if canonical_export:
        result[PREFIX+"DETAIL_TO_BOUNDARY_CANONICAL_EXPORT_CSV"]=suffix_env["DETAIL_TO_BOUNDARY_CANONICAL_EXPORT_CSV"]
    else:
        result.pop(PREFIX+"DETAIL_TO_BOUNDARY_CANONICAL_EXPORT_CSV",None)
    forbidden=("NUMI_HUMAN_SKIN_FAT_CANDIDATE033_SOURCE_PREFLIGHT",
               "NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_CANDIDATE_SHA256",
               "NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_CANDIDATE_ID",
               "NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_DIAGNOSTIC")
    for key in forbidden:
        if key in result: raise ValueError(f"stale candidate-specific env is forbidden: {key}")
    return result

def build_plan(config: dict, outdir: Path, source_preflight_receipt: dict|None=None, *, require_fresh_outputs: bool=True) -> dict:
    if config.get("schema")!="numi.human.h706.native-launch-inputs.v1":
        raise ValueError("unsupported route manifest schema")
    static_pins=verify_pinned_static()
    pins=verify_input_pins(config.get("input_pins",{}))
    geo,_=check_claim_receipt(config.get("geometry_admission",{}),REQUIRED_GEOMETRY_CLAIMS,"geometry admission")
    mass,_=check_claim_receipt(config.get("mass_admission",{}),REQUIRED_MASS_CLAIMS,"mass admission")
    suffix_env=validate_environment(config.get("native_preflight_env",{}),pins,
                                    config.get("detail_map_export_target",""))
    scene=config.get("scene",{})
    argv=scene.get("argv")
    scene_env=scene.get("environment",{})
    if not isinstance(argv,list) or not all(isinstance(x,str) for x in argv):
        raise ValueError("scene.argv must be the exact existing native probe argv without executable")
    validate_scene_output_slot(argv)
    validate_full_scene_mode(argv,scene_env)
    expected_steps=argv_step_count(argv)
    if "--persistent-metal-stand" not in argv:
        raise ValueError("motion argv must preserve the persistent accepted-state standing route")
    if not isinstance(scene_env,dict): raise ValueError("scene.environment must be a key/value map")
    if scene_env.get("NUMI_HUMAN_FEM_SKIN_SUPPORT") == "1" and suffix_env.get(
            "MAX_PARENT_REGISTRATION_DISTANCE_M") != MAX_NATIVE_PARENT_REGISTRATION_M:
        raise ValueError("FEM skin support must use the current 0.0004 m registration bound")
    output_tokens={"{RUN_OUTPUT}","{RUN_MOVIE}"}
    for i,arg in enumerate(argv):
        if arg in output_tokens:
            continue
        if arg.startswith("/") and arg not in pins:
            raise ValueError(f"scene input is not pinned: {arg}")
    for key,value in scene_env.items():
        if isinstance(value,str) and value.startswith("/") and Path(value).is_file() and value not in pins:
            raise ValueError(f"scene input environment path {key} is not SHA-pinned: {value}")
    output_root=Path(config.get("output_root",""))
    if not str(output_root).startswith("/"):
        raise ValueError("output_root must be a new absolute task-owned path")
    if require_fresh_outputs and output_root.exists():
        raise ValueError(f"output_root already exists: {output_root}")
    export_path=config["detail_map_export_target"]
    if export_path in pins:
        raise ValueError("canonical detail map output must not be confused with an input pin")
    if require_fresh_outputs and Path(export_path).exists():
        raise ValueError(f"canonical detail map output already exists: {export_path}")
    pre_env=build_stage_env(scene_env,suffix_env,source_only=True,canonical_export=True,
                            operational_split=True)
    pre_env=expand_stage_environment(pre_env,output_root/"source-preflight")
    pre_argv=[str(BINARY)]+expand_scene_argv(argv,output_root/"source-preflight")
    plan={
      "schema":"numi.human.h706.native-launch-plan.v1",
      "status":"SOURCE_PREFLIGHT_READY",
      "execution_performed":False,"native_runtime_executed":False,"GPU_executed":False,
      "native_binary":{"path":str(BINARY),"sha256":static_pins[str(BINARY)],
                       "capacity_profile":"full-reference-500k-nodes-2m-cells",
                       "compiled_vfs_content_sha256":{p:static_pins[p] for p in VFS_CONTENT_PINS},
                       "runtime_dylib_metallib_sha256":{p:static_pins[p] for p in RUNTIME_ASSET_PINS}},
      "source330":{"packet_csv":str(PARENT),"packet_sha256":PINNED_PARENT[str(PARENT)],
                   "labels_csv":str(PARENT)+".labels.csv",
                   "labels_sha256":PINNED_PARENT[str(PARENT)+".labels.csv"],
                   "pose_snapshot":str(SOURCE330/"registered-skin-poses-100.json"),
                   "pose_sha256":PINNED_PARENT[str(SOURCE330/"registered-skin-poses-100.json")]},
      "geometry_admission":{"receipt_path":config["geometry_admission"]["path"],
                            "receipt_sha256":config["geometry_admission"]["sha256"],
                            "claims":config["geometry_admission"]["claims"]},
      "mass_admission":{"receipt_path":config["mass_admission"]["path"],
                        "receipt_sha256":config["mass_admission"]["sha256"],
                        "claims":config["mass_admission"]["claims"]},
      "input_pins":pins,
      "physics_environment_sha256":hashlib.sha256(json.dumps(scene_env,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
      "stages":[{"name":"source_preflight_and_detail_map_export","argv":pre_argv,
                 "environment":pre_env,"expected_runtime":False,"expected_accepted_steps":0,
                 "canonical_detail_map_output":export_path}],
      "motion":{"requested_accepted_step_count":expected_steps,
                "install_and_motion_are_same_probe_invocation":True,
                "source_only_reference_domain":"candidateOperationalReference",
                "physical_reference_domain":"candidateOperationalReference_with_verified_conservative_spatial_install",
                "historical_mass_source_domain":"sourceMassParent",
                "postrun_admission_required":True,
                "postrun_acceptance_requirements":[
                    "whole_body_skin_fat_registered_volume_install=PASS",
                    "resting_FEM_terminal_audit=passed",
                    "accepted_steps=requested_accepted_step_count",
                    "geometry_and_mass_admission_receipts_remain_hash_valid"],
                "physical_admission_is_upstream_receipt_gate":True},
    }
    if source_preflight_receipt is not None:
        expected_cfg=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        if source_preflight_receipt.get("status")!="PASS_SOURCE_ONLY_NO_RUNTIME":
            raise ValueError("source-preflight receipt is not a passing zero-runtime result")
        if source_preflight_receipt.get("input_manifest_sha256")!=expected_cfg:
            raise ValueError("source-preflight receipt binds a different H706 manifest")
        detail_path=source_preflight_receipt.get("canonical_detail_map_path")
        detail_sha=source_preflight_receipt.get("canonical_detail_map_sha256")
        if detail_path != config.get("detail_map_export_target"):
            raise ValueError("source-preflight receipt names a different canonical detail-map output")
        checked_hash(detail_path,detail_sha,"source-preflight canonical detail map")
        motion_env=suffix_env.copy()
        motion_env["DETAIL_TO_BOUNDARY_CSV"]=detail_path
        motion_env["DETAIL_TO_BOUNDARY_SHA256"]=detail_sha
        motion_env.pop("DETAIL_TO_BOUNDARY_CANONICAL_EXPORT_CSV",None)
        # The physical stage explicitly opts into the authenticated spatial
        # operational-reference installer. Source330 historical registration
        # remains separately pinned; conservative spatial mass mapping and its
        # existing distance/density gates remain mandatory.
        final_env=build_stage_env(scene_env,motion_env,source_only=False,
                                  canonical_export=False,operational_split=True,
                                  operational_spatial_install=True)
        final_env=expand_stage_environment(final_env,output_root/"install-motion")
        validate_physical_spatial_install(final_env)
        validate_full_scene_mode(argv,final_env)
        motion_argv=[str(BINARY)]+expand_scene_argv(argv,output_root/"install-motion")
        plan["status"]="MOTION_LAUNCH_READY_AFTER_SOURCE_PREFLIGHT"
        plan["source_preflight_receipt"]=source_preflight_receipt
        plan["stages"].append({"name":"physical_install_and_accepted_state_motion",
                               "argv":motion_argv,"environment":final_env,
                               "expected_runtime":True,"expected_accepted_steps":expected_steps,
                               "note":"The existing probe installs the registered volume before stepping; it exposes no separate install-only entry point."})
    return plan

def record_source_preflight(plan_path: Path, stdout_path: Path, stderr_path: Path,
                            exit_code: int, out_path: Path) -> dict:
    plan=read(plan_path)
    if plan.get("status")!="SOURCE_PREFLIGHT_READY":
        raise ValueError("source plan is not ready for preflight")
    verify_saved_plan_inputs(plan)
    if exit_code != 0:
        raise ValueError(f"native source-only preflight exited with {exit_code}")
    stdout=stdout_path.read_text(errors="replace")
    stderr=stderr_path.read_text(errors="replace")
    if "registered_volume_native_preflight=PASS" not in stdout:
        raise ValueError("native source-only preflight PASS marker missing from stdout")
    if "registered_volume_detail_canonical_export_source_only=PASS" not in stdout:
        raise ValueError("691 canonical detail-face export PASS marker missing")
    if "physical_admission=0" not in stdout or "accepted_steps=0" not in stdout:
        raise ValueError("source-only output must explicitly show no admission and zero accepted steps")
    require_no_failure_markers(stdout,stderr,(
        "registered_volume_native_preflight=FAIL",
        "registered_volume_detail_canonical_export_source_only=FAIL",
    ),"source-only preflight")
    export_path=plan["stages"][0]["canonical_detail_map_output"]
    export_sha=sha256(Path(export_path))
    match=re.search(r"registered_volume_detail_canonical_export_source_only=PASS[^\n]*output_sha=([0-9a-f]{64})",stdout)
    if not match or match.group(1)!=export_sha:
        raise ValueError("canonical detail map output hash does not match the native export receipt line")
    receipt={
       "schema":"numi.human.h706.native-source-preflight-receipt.v1",
       "status":"PASS_SOURCE_ONLY_NO_RUNTIME",
       "input_manifest_sha256":plan["manifest_sha256"],
       "native_binary_sha256":plan["native_binary"]["sha256"],
       "stdout_sha256":sha256(stdout_path),"stderr_sha256":sha256(stderr_path),
       "canonical_detail_map_path":export_path,"canonical_detail_map_sha256":export_sha,
       "physical_admission":False,"runtime_created":False,"accepted_steps":0,
       "execution_performed":True,
    }
    dump(out_path,receipt)
    return receipt

def record_motion(plan_path: Path, stdout_path: Path, stderr_path: Path,
                  exit_code: int, out_path: Path) -> dict:
    plan=read(plan_path)
    if plan.get("status")!="MOTION_LAUNCH_READY_AFTER_SOURCE_PREFLIGHT":
        raise ValueError("motion plan is not ready after source preflight")
    verify_saved_plan_inputs(plan)
    source_receipt_path=plan.get("source_preflight_receipt_path")
    source_receipt_sha=plan.get("source_preflight_receipt_sha256")
    if not isinstance(source_receipt_path,str) or not isinstance(source_receipt_sha,str):
        raise ValueError("motion plan lacks its source-only receipt pin")
    checked_hash(source_receipt_path,source_receipt_sha,"source-only receipt")
    source_receipt=read(Path(source_receipt_path))
    if source_receipt.get("status")!="PASS_SOURCE_ONLY_NO_RUNTIME" or source_receipt.get("input_manifest_sha256")!=plan.get("manifest_sha256"):
        raise ValueError("motion plan source-only receipt is stale or bound to another manifest")
    if source_receipt != plan.get("source_preflight_receipt"):
        raise ValueError("motion plan embeds a different source-only receipt than its pinned file")
    expected_detail=plan["stages"][0]["canonical_detail_map_output"]
    if source_receipt.get("canonical_detail_map_path")!=expected_detail:
        raise ValueError("source-only receipt names a different canonical detail map")
    checked_hash(source_receipt["canonical_detail_map_path"],
                 source_receipt["canonical_detail_map_sha256"],
                 "canonical detail-map output")
    motion_env=plan["stages"][1]["environment"]
    if (motion_env.get(PREFIX+"DETAIL_TO_BOUNDARY_CSV")!=source_receipt["canonical_detail_map_path"] or
            motion_env.get(PREFIX+"DETAIL_TO_BOUNDARY_SHA256")!=source_receipt["canonical_detail_map_sha256"]):
        raise ValueError("physical stage is not bound to the source-only canonical detail map")
    if exit_code != 0:
        raise ValueError(f"native motion process exited with {exit_code}")
    stdout=stdout_path.read_text(errors="replace")
    stderr=stderr_path.read_text(errors="replace")
    if "whole_body_skin_fat_registered_volume_install=PASS" not in stdout:
        raise ValueError("registered-volume installation PASS marker missing")
    require_no_failure_markers(stdout,stderr,(
        "whole_body_skin_fat_registered_volume_install=FAIL",
        "registered_volume_native_preflight=FAIL",
        "registered_volume_detail_canonical_export_source_only=FAIL",
    ),"native motion")
    requested=int(plan["motion"]["requested_accepted_step_count"])
    pattern=(r"resting_FEM_terminal_audit=passed[^\n]*expected_steps="+
             str(requested)+r"[^\n]*accepted_steps="+str(requested)+r"(?:\s|$)")
    if not re.search(pattern,stdout):
        raise ValueError("terminal audit does not attest the requested accepted-state step count")
    if "resting_FEM_terminal_audit=failed" in stdout+stderr:
        raise ValueError("native log contains a failed terminal audit")
    receipt={
       "schema":"numi.human.h706.native-motion-receipt.v1",
       "status":"PASS_INSTALL_AND_ACCEPTED_STATE_MOTION",
       "plan_sha256":sha256(plan_path),
       "stdout_sha256":sha256(stdout_path),"stderr_sha256":sha256(stderr_path),
       "native_exit_code":exit_code,
       "native_binary_sha256":plan["native_binary"]["sha256"],
       "geometry_admission_receipt_sha256":plan["geometry_admission"]["receipt_sha256"],
       "mass_admission_receipt_sha256":plan["mass_admission"]["receipt_sha256"],
       "source_preflight_receipt_sha256":source_receipt_sha,
       "canonical_detail_map_sha256":source_receipt["canonical_detail_map_sha256"],
       "registered_volume_install_pass":True,
       "requested_accepted_steps":requested,"observed_accepted_steps":requested,
       "native_runtime_executed":True,"execution_performed":True,
       "physical_admission_receipt_revalidated":True,
    }
    dump(out_path,receipt)
    return receipt

def render_shell(stage: dict) -> str:
    pairs=[]
    for key,value in sorted(stage["environment"].items()):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",key):
            raise ValueError(f"invalid environment key {key}")
        pairs.append(f"{key}={shlex.quote(str(value))}")
    return "#!/bin/sh\nset -eu\nexec env " + " ".join(pairs) + " " + shlex.join(stage["argv"]) + "\n"

def readiness(manifest_path: Path) -> dict:
    static=verify_pinned_static()
    parent={k:checked_hash(k,v,"source330") for k,v in PINNED_PARENT.items()}
    config=read(manifest_path)
    blockers=[]
    input_pin_errors=input_pin_diagnostics(config.get("input_pins",{}))
    input_pin_error=input_pin_errors[0] if input_pin_errors else None
    if input_pin_errors:
        blockers.append("one or more current Human scene input pins are missing or changed")
    if config.get("schema")!="numi.human.h706.native-launch-inputs.v1":
        blockers.append("route input schema missing or invalid")
    native_env=config.get("native_preflight_env",{})
    if not isinstance(native_env,dict):
        native_env={}
    missing_bundle_fields=sorted(REQUIRED_ENV-set(native_env))
    if missing_bundle_fields:
        blockers.append("current O/I/D and 217 volume/registration bundle inputs are incomplete")
    if not config.get("geometry_admission"): blockers.append("candidate-specific geometry admission receipt is absent")
    if not config.get("mass_admission"): blockers.append("candidate-specific conservative mass/density admission receipt is absent")
    if not config.get("scene",{}).get("argv"): blockers.append("current Human scene argv is absent")
    return {"schema":"numi.human.h706.native-launch-readiness.v1",
            "status":"WAITING_FOR_CURRENT_228_BUNDLE_AND_ADMISSION_RECEIPTS" if blockers else "INPUTS_PRESENT",
            "execution_performed":False,"native_runtime_executed":False,"GPU_executed":False,
            "native721_sha256":static[str(BINARY)],
            "compiled_probe_source_sha256":static["__compiled_probe_source_sha256__"],
            "native721_vfs_external_content_count":int(static["__vfs_external_content_count__"]),
            "native721_vfs_external_content":static["__vfs_external_content__"],
            "native721_runtime_dylib_metallib_sha256":{p:static[p] for p in RUNTIME_ASSET_PINS},
            "source330_file_count":len(parent),
            "source330_hashes":parent,"blockers":blockers,
            "candidate_registered_bundle_missing_fields":missing_bundle_fields,
            "scene_input_pin_error":input_pin_error,
            "scene_input_pin_errors":input_pin_errors,
            "route_contract":{"source_preflight":"source-only with registered import and 691 canonical detail-map export",
                "install_motion":"explicit operational-spatial install, then accepted-state stepping in one existing probe invocation",
                "mass_transport_max_m":MAX_MASS_TRANSPORT_M,
                "native_parent_registration_max_m":MAX_NATIVE_PARENT_REGISTRATION_M,
                "legacy_identity_or_530_chain_claim":False}}

def main() -> int:
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="cmd",required=True)
    g=sub.add_parser("generate",help="write source-preflight plan and, after a receipt, install+motion plan")
    g.add_argument("--manifest",required=True); g.add_argument("--out",required=True)
    g.add_argument("--source-preflight-receipt")
    ready=sub.add_parser("readiness",help="verify frozen native721/source330 and report waiting inputs")
    ready.add_argument("--manifest",required=True); ready.add_argument("--out",required=True)
    r=sub.add_parser("record-source-preflight",help="verify saved zero-runtime stdout and canonical face-map output")
    r.add_argument("--plan",required=True); r.add_argument("--stdout",required=True); r.add_argument("--stderr",required=True); r.add_argument("--exit-code",type=int,required=True); r.add_argument("--out",required=True)
    m=sub.add_parser("record-motion",help="verify installed volume and accepted-state terminal audit from saved native logs")
    m.add_argument("--plan",required=True); m.add_argument("--stdout",required=True); m.add_argument("--stderr",required=True); m.add_argument("--exit-code",type=int,required=True); m.add_argument("--out",required=True)
    a=ap.parse_args()
    try:
        if a.cmd=="readiness":
            report=readiness(Path(a.manifest))
            dump(Path(a.out),report)
            print(json.dumps({"status":report["status"],"report":a.out,
                              "blockers":report["blockers"],"execution_performed":False},sort_keys=True))
            return 0
        if a.cmd=="record-source-preflight":
            receipt=record_source_preflight(Path(a.plan),Path(a.stdout),Path(a.stderr),a.exit_code,Path(a.out))
            print(json.dumps({"status":receipt["status"],"receipt":a.out,"sha256":sha256(Path(a.out))},sort_keys=True))
            return 0
        if a.cmd=="record-motion":
            receipt=record_motion(Path(a.plan),Path(a.stdout),Path(a.stderr),a.exit_code,Path(a.out))
            print(json.dumps({"status":receipt["status"],"receipt":a.out,"sha256":sha256(Path(a.out))},sort_keys=True))
            return 0
        config=read(Path(a.manifest))
        plan=build_plan(config,Path(a.out),read(Path(a.source_preflight_receipt)) if a.source_preflight_receipt else None)
        plan["manifest_sha256"]=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        plan["manifest_path"]=str(Path(a.manifest).resolve())
        if a.source_preflight_receipt:
            receipt_path=Path(a.source_preflight_receipt).resolve()
            plan["source_preflight_receipt_path"]=str(receipt_path)
            plan["source_preflight_receipt_sha256"]=sha256(receipt_path)
        output=Path(a.out)
        stage_paths=[output.with_name(f"{output.stem}-stage-{i:02d}.sh") for i in range(1,len(plan["stages"])+1)]
        collisions=[str(path) for path in [output,*stage_paths] if path.exists()]
        if collisions: raise ValueError(f"generated plan output already exists: {collisions}")
        dump(Path(a.out),plan)
        for index,stage in enumerate(plan["stages"],1):
            sh=Path(a.out).with_name(f"{Path(a.out).stem}-stage-{index:02d}.sh")
            sh.write_text(render_shell(stage)); sh.chmod(0o755)
        print(json.dumps({"status":plan["status"],"plan":a.out,
                          "stages":[s["name"] for s in plan["stages"]],
                          "execution_performed":False},sort_keys=True))
        return 0
    except Exception as e:
        print(f"BLOCKED: {e}",file=sys.stderr)
        return 2

if __name__=="__main__": raise SystemExit(main())
