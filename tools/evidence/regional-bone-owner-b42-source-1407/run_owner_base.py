from __future__ import annotations
import hashlib, json, signal, sys, time, traceback
from pathlib import Path

BASE = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/current-manifest-recovery-004")
REPO = Path("/Users/n/numi-human-final-integration-publish-1397")
SOURCES = BASE / "fit-source-root-001"
SOURCE_ROOT_MANIFEST = BASE / "fit-source-root-001.json"
REGISTRATION = BASE / "bilateral-regional-b42-current-source-compatibility-001/candidate-current-source.json"
COMPATIBILITY = BASE / "bilateral-regional-b42-current-source-compatibility-001/compatibility-report.json"
ARTIFACT = BASE / "recovered-artifact"
OUT = BASE / "bilateral-b42-native-bone-owner-001"
OWNER = REPO / "src/numilab_human/model.py"
SURFACE_MAP = Path("/Users/n/numi-human-anatomy-finish-1378/config/bodyparts3d-myosim-surface-map.v1.json")

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def digest_tree(path: Path) -> dict:
    rows = []
    for p in sorted(path.rglob("*")):
        if p.is_file():
            rows.append({"path": str(p.relative_to(path)), "sha256": sha(p), "bytes": p.stat().st_size})
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"file_count": len(rows), "total_bytes": sum(x["bytes"] for x in rows),
            "sha256": hashlib.sha256(blob).hexdigest(), "files": rows}

def readj(path):
    return json.loads(path.read_text(encoding="utf-8"))

assert OUT.is_dir() and not (OUT/"compile-report.json").exists(), "output must be fresh and precreated"
REPORT = OUT / "compile-report.json"
record = {"schema":"numi.human.upper-limb-full-owner-base-compile.v1",
          "status":"preparing", "phase":"preflight",
          "started_unix":time.time(), "output":str(OUT),
          "scope":"Existing-owner 185-bone source compile bound to regional registration31c77660; no geometry or native admission."}
def save():
    REPORT.write_text(json.dumps(record, indent=2, sort_keys=True)+"\n", encoding="utf-8")
def alarm_handler(signum, frame):
    raise TimeoutError("20-minute source owner compile budget exceeded")
signal.signal(signal.SIGALRM, alarm_handler)
signal.alarm(1200)
try:
    cand_sha = sha(REGISTRATION)
    compat_sha = sha(COMPATIBILITY)
    root_sha = sha(SOURCE_ROOT_MANIFEST)
    root = readj(SOURCE_ROOT_MANIFEST)
    cand = readj(REGISTRATION)
    compat = readj(COMPATIBILITY)
    artifact_manifest_path = ARTIFACT / "myosim-fullbody-reference.manifest.json"
    artifact_manifest = readj(artifact_manifest_path)
    bodyparts = cand["source"]["bodyparts"]
    assert cand_sha == "31c77660a16e8688fe68489eadff9d3861ceb27ed3b73d0288bda330d2a13b8c"
    assert compat_sha == "f70ac7ced0b1fbcc6186c24f67aab47461425d5f76c7a191903050552293190e"
    assert bodyparts["id"] == "bodyparts3d_4" and bodyparts["version"] == "4.0"
    source_descriptors = {x["name"]:x["sha256"] for x in root["bodyparts_sources"]}
    assert bodyparts["archives"] == [
      {"file":"isa_BP3D_4.0_obj_99.zip","hierarchy":"is_a","sha256":source_descriptors["isa_BP3D_4.0_obj_99.zip"]},
      {"file":"partof_BP3D_4.0_obj_99.zip","hierarchy":"part_of","sha256":source_descriptors["partof_BP3D_4.0_obj_99.zip"]},
    ]
    assert cand["source"]["myosim"]["artifact_manifest_sha256"] == sha(artifact_manifest_path)
    assert cand["source"]["myosim"]["source"]["archive_sha256"] == artifact_manifest["source"]["archive_sha256"]
    assert cand["source"]["myosim"]["payloads"]["muscles"]["sha256"] == sha(ARTIFACT/"myosim-fullbody-muscle-reference.nhmyo")
    assert cand["source"]["myosim"]["payloads"]["rigid"]["sha256"] == sha(ARTIFACT/"myosim-fullbody-core-reference.nhrigid")
    assert cand["source"]["myosim"]["payloads"]["muscles"]["sha256"] == "e5bb8a8168706bb3b23cf42b1d849659569ba2e6027bcc85729c3110fdbb8c4b"
    assert cand["source"]["myosim"]["payloads"]["rigid"]["sha256"] == "2c78cb4150b97cea6e8169dad9e8f5dd59af667b247e07b56e48e857435560e4"
    source_file_rows = []
    for item in root["bodyparts_sources"]:
        p = Path(item["source"]).resolve()
        assert p.is_file() and sha(p) == item["sha256"], f"source input drift: {p}"
        local = SOURCES / item["name"]
        assert local.is_file() and sha(local) == item["sha256"], f"source-root input drift: {local}"
        source_file_rows.append({"name":item["name"], "resolved_path":str(p), "sha256":item["sha256"], "bytes":p.stat().st_size})
    assert sha(SURFACE_MAP) == "1d4bbbf216e1a67ac7e27f87d61cbda513f5661b703c5c450c05a51c3c31caa2"
    myosim_tree_before = digest_tree(SOURCES/"myosim")
    assert myosim_tree_before["file_count"] > 200
    input_paths = [REGISTRATION, COMPATIBILITY, SOURCE_ROOT_MANIFEST, OWNER, SURFACE_MAP,
                   artifact_manifest_path, ARTIFACT/"myosim-fullbody-core-reference.nhrigid",
                   ARTIFACT/"myosim-fullbody-muscle-reference.nhmyo"]
    input_before = {str(p.resolve()):sha(p.resolve()) for p in input_paths}
    record.update({
      "phase":"preflight_passed",
      "candidate_path":str(REGISTRATION), "candidate_sha256":cand_sha,
      "compatibility_proof_path":str(COMPATIBILITY), "compatibility_proof_sha256":compat_sha,
      "source_root_manifest_path":str(SOURCE_ROOT_MANIFEST), "source_root_manifest_sha256":root_sha,
      "bodyparts_source_inputs":source_file_rows,
      "myosim_source_tree":{k:v for k,v in myosim_tree_before.items() if k!="files"},
      "myosim_source_tree_file_manifest":myosim_tree_before["files"],
      "artifact_manifest_sha256":sha(artifact_manifest_path),
      "nhrigid_sha256":sha(ARTIFACT/"myosim-fullbody-core-reference.nhrigid"),
      "nhmyo_sha256":sha(ARTIFACT/"myosim-fullbody-muscle-reference.nhmyo"),
      "owner_module_path":str(OWNER), "owner_module_sha256":sha(OWNER),
      "surface_map_path":str(SURFACE_MAP), "surface_map_sha256":sha(SURFACE_MAP),
      "input_hashes_before":input_before,
      "owner":"numilab_human.model.bodyparts_myosim_bone_visual_payload",
      "owner_call":"Direct library invocation with anatomy identity copied from candidate source.bodyparts; the CLI parser's broader label/graph inventory is not needed by this owner, which validates exact archive/member IDs and uses both pinned element maps.",
      "compatibility_target_frame_rows":compat["candidate_target_body_frames"]["proof_rows"],
      "admission":False,
    })
    save()
    phase="owner_compile"
    record["phase"]=phase
    save()
    sys.path.insert(0, str(REPO/"src"))
    from numilab_human.model import bodyparts_myosim_bone_visual_payload
    anatomy = {"source_id":bodyparts["id"], "version":bodyparts["version"], "archives":bodyparts["archives"]}
    manifest = bodyparts_myosim_bone_visual_payload(
        SOURCES, anatomy, REGISTRATION, OUT/"owner-base", artifact=ARTIFACT
    )
    phase="postcompile_checks"
    record["phase"]=phase
    save()
    payload = OUT/"owner-base"/manifest["payload"]["file"]
    manifest_path = OUT/"owner-base"/"bodyparts3d-myosim-major-bones.manifest.json"
    generated = readj(manifest_path)
    assert sha(payload)==generated["payload"]["sha256"]
    assert generated["source"]["registration"]["sha256"]==cand_sha
    assert generated["source"]["bodyparts"]==bodyparts
    assert generated["payload"]["bone_count"]==185, generated["payload"]
    ids=list(range(1,186))
    record.update({
      "status":"compiled_185_bone_owner_base_unadmitted",
      "phase":"passed",
      "finished_unix":time.time(),
      "owner_base_payload_path":str(payload),
      "owner_base_payload_sha256":sha(payload),
      "owner_base_manifest_path":str(manifest_path),
      "owner_base_manifest_sha256":sha(manifest_path),
      "coverage":generated["coverage"],
      "surface_stable_ids":ids,
      "source_provenance":generated["source"],
      "input_hashes_after":{str(p.resolve()):sha(p.resolve()) for p in input_paths},
      "myosim_source_tree_after":{k:v for k,v in digest_tree(SOURCES/"myosim").items() if k!="files"},
      "evidence_boundary":"Existing NHBONES owner compile only; registration and source pinned, anatomy and native admission pending."
    })
    assert record["input_hashes_after"]==input_before
    assert record["myosim_source_tree_after"]==record["myosim_source_tree"]
    save()
except BaseException as e:
    record.update({"status":"failed", "phase":locals().get("phase", record.get("phase")),
                   "error":repr(e), "traceback":traceback.format_exc(), "finished_unix":time.time()})
    save()
    raise
finally:
    signal.alarm(0)

