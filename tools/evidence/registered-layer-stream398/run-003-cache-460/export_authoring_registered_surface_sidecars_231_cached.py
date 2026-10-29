#!/usr/bin/env python3
"""Hash-pinned O/I role-cache variant of the frozen 398 sidecar exporter.

Existing 398 source is imported unchanged. Cached O/I CSVs are copied only when the
candidate, parent, pose, full86, producer and cache receipt pins all match.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, os, shutil, sys
from pathlib import Path
from typing import Any

ENGINE_DIR = Path(__file__).resolve().parents[1] / "run-002"
if not ENGINE_DIR.is_dir():
    raise RuntimeError(f"frozen 398 engine directory is missing: {ENGINE_DIR}")
sys.path.insert(0, str(ENGINE_DIR))
import export_authoring_registered_surface_sidecars_230 as engine  # noqa: E402

ROLE_FILES = {
    "O": ("registered-O.csv", "registered-O-faces.csv",
          "parent-tet-full-weights-O.csv", "parent-tet-full-weight-owners-O.csv"),
    "I": ("registered-I.csv", "registered-I-faces.csv",
          "parent-tet-full-weights-I.csv", "parent-tet-full-weight-owners-I.csv"),
}
PARENT_HASH_FIELDS = {
    "mass": ("massSHA256", ""),
    "node": ("nodeSHA256", ".nodes.csv"),
    "source": ("sourceSHA256", ".source.csv"),
    "tets": ("tetsSHA256", ".tets.csv"),
    "labels": ("labelsSHA256", ".labels.csv"),
}

class CacheAdmissionError(ValueError):
    pass

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def require_equal(label: str, expected: Any, actual: Any) -> None:
    if expected != actual:
        raise CacheAdmissionError(
            f"cached O/I input mismatch for {label}: expected {expected!r}, got {actual!r}"
        )

def option_value(argv: list[str], name: str, default: str | None = None) -> str | None:
    for index, item in enumerate(argv[:-1]):
        if item == name:
            return argv[index + 1]
    return default

def cache_artifact_hashes(cache_dir: Path, receipt: dict[str, Any]) -> dict[str, str]:
    outputs = receipt.get("outputSHA256")
    if not isinstance(outputs, dict):
        raise CacheAdmissionError("cached receipt has no outputSHA256 map")
    names = [name for role in ("O", "I") for name in ROLE_FILES[role]]
    names.extend(("node-parent-registration.csv", "donor-frames.csv"))
    fallback = receipt.get("exactParentNearTieFallbackReports", {})
    for role in ("O", "I"):
        report = fallback.get(role)
        if report:
            names.append(report["path"])
    verified = {}
    for name in names:
        expected = outputs.get(name)
        path = cache_dir / name
        if not isinstance(expected, str) or not path.is_file():
            raise CacheAdmissionError(f"cached O/I artifact is missing or unpinned: {path}")
        actual = sha256_file(path)
        require_equal(f"cached artifact {name}", expected, actual)
        verified[name] = actual
    return verified

def copy_cached_role_artifacts(cache_dir: Path, destination: Path,
                               receipt: dict[str, Any],
                               roles: tuple[str, ...] = ("O", "I")) -> dict[str, str]:
    verified = cache_artifact_hashes(cache_dir, receipt)
    destination.mkdir(parents=True, exist_ok=True)
    copied = {}
    for role in roles:
        if role not in ROLE_FILES:
            raise CacheAdmissionError(f"unsupported cached role {role}")
        for name in ROLE_FILES[role]:
            target = destination / name
            shutil.copyfile(cache_dir / name, target)
            actual = sha256_file(target)
            require_equal(f"copied artifact {name}", verified[name], actual)
            copied[name] = actual
    return copied

def validate_cache_inputs(args: argparse.Namespace) -> dict[str, Any]:
    needed = (args.outer, args.inner, args.deep, args.parent, args.poses,
              args.parent_full86_weights, args.reuse_oi_cache_dir,
              args.reuse_oi_cache_launch)
    if any(value is None for value in needed):
        raise CacheAdmissionError("cached O/I reuse requires explicit O/I/D, parent, pose, weights, cache and launch paths")
    cache_dir = Path(args.reuse_oi_cache_dir)
    receipt_path = cache_dir / "receipt.json"
    launch_path = Path(args.reuse_oi_cache_launch)
    if not receipt_path.is_file() or not launch_path.is_file():
        raise CacheAdmissionError("pinned cache receipt or producer launch manifest is missing")
    receipt_sha = sha256_file(receipt_path)
    launch_sha = sha256_file(launch_path)
    require_equal("cache receipt SHA-256 pin", args.reuse_oi_cache_receipt_sha256, receipt_sha)
    require_equal("cache launch SHA-256 pin", args.reuse_oi_cache_launch_sha256, launch_sha)
    receipt = json.loads(receipt_path.read_text())
    launch = json.loads(launch_path.read_text())
    if receipt.get("schema") != "registered-authoring-surface-sidecars-receipt.v1":
        raise CacheAdmissionError("cache receipt schema is not the supported sidecar format")
    if receipt.get("status") != "reference_registration_ready_geometry_not_admitted":
        raise CacheAdmissionError("cache receipt is not a complete registration-only result")
    if receipt.get("physical_admission") is not False or receipt.get("geometry_admission") is not False:
        raise CacheAdmissionError("cache receipt contains an unexpected admission claim")
    if receipt.get("native_steps") != 0:
        raise CacheAdmissionError("cache receipt has nonzero native step count")

    engine_path = Path(engine.__file__).resolve()
    engine_sha = sha256_file(engine_path)
    require_equal("frozen 398 exporter source SHA-256 pin", args.reuse_oi_engine_sha256, engine_sha)
    require_equal("producer launch exporter source SHA-256", args.reuse_oi_engine_sha256, launch.get("source_sha256"))
    argv = launch.get("argv")
    if not isinstance(argv, list):
        raise CacheAdmissionError("cache launch manifest has no argv list")
    launched_engine = next((Path(v).resolve() for v in argv
        if str(v).endswith("export_authoring_registered_surface_sidecars_230.py")), None)
    require_equal("producer launch exporter path", engine_path, launched_engine)

    candidates = receipt.get("candidateSurfaces", {})
    for role, current, option in (("O", Path(args.outer), "--outer"),
                                   ("I", Path(args.inner), "--inner")):
        record = candidates.get(role)
        if not isinstance(record, dict):
            raise CacheAdmissionError(f"cache receipt is missing candidate role {role}")
        require_equal(f"cache launch {role} source path", record.get("path"), option_value(argv, option))
        expected = record.get("sha256")
        require_equal(f"{role} candidate input SHA-256", expected, sha256_file(current))
        source_path = Path(record["path"])
        if source_path.is_file():
            require_equal(f"cached source {role} candidate SHA-256", expected, sha256_file(source_path))

    parent = receipt.get("parentCurrent", {})
    require_equal("cache launch parent path", parent.get("massPath"), option_value(argv, "--parent"))
    parent_base = Path(args.parent)
    for label, (field, suffix) in PARENT_HASH_FIELDS.items():
        expected = parent.get(field)
        if not isinstance(expected, str):
            raise CacheAdmissionError(f"cache receipt is missing parent {field}")
        path = parent_base if not suffix else Path(str(parent_base) + suffix)
        if not path.is_file():
            raise CacheAdmissionError(f"current parent {label} input is missing: {path}")
        require_equal(f"parent {label} SHA-256", expected, sha256_file(path))

    pose = receipt.get("poseSnapshot", {})
    require_equal("cache launch pose path", pose.get("path"), option_value(argv, "--poses"))
    require_equal("pose snapshot SHA-256", pose.get("sha256"), sha256_file(Path(args.poses)))
    weights = receipt.get("parentFull86Weights", {})
    require_equal("cache launch parent full86 path", weights.get("path"),
                  option_value(argv, "--parent-full86-weights"))
    require_equal("parent full86 weights SHA-256", weights.get("sha256"),
                  sha256_file(Path(args.parent_full86_weights)))
    require_equal("full86 parent-node SHA-256", parent.get("nodeSHA256"),
                  weights.get("parentNodesSHA256"))

    limits = receipt.get("limits", {})
    old_max = float(option_value(argv, "--maximum-parent-distance-m", "0.02"))
    old_tie = float(option_value(argv, "--tie-tolerance-m", "1e-8"))
    require_equal("recorded maximum parent distance",
                  limits.get("maximumParentRegistrationDistanceM"), old_max)
    require_equal("recorded tie tolerance", limits.get("tieToleranceM"), old_tie)
    require_equal("requested maximum parent distance", old_max,
                  float(args.maximum_parent_distance_m))
    require_equal("requested tie tolerance", old_tie, float(args.tie_tolerance_m))
    require_equal("material-by-role policy", engine.MATERIAL, limits.get("materialBySurface"))

    # D is intentionally a new input and is recorded, but it is not part of the O/I cache key.
    d_sha = sha256_file(Path(args.deep))
    artifacts = cache_artifact_hashes(cache_dir, receipt)
    return {
        "cacheReceiptPath": str(receipt_path), "cacheReceiptSHA256": receipt_sha,
        "cacheLaunchPath": str(launch_path), "cacheLaunchSHA256": launch_sha,
        "enginePath": str(engine_path), "engineSHA256": engine_sha,
        "cacheVariantSHA256": sha256_file(Path(__file__).resolve()),
        "candidateSHA256": {"O": sha256_file(Path(args.outer)),
                            "I": sha256_file(Path(args.inner)), "D": d_sha},
        "parentSHA256": {label: parent[field] for label, (field, _) in PARENT_HASH_FIELDS.items()},
        "poseSHA256": pose["sha256"], "parentFull86WeightsSHA256": weights["sha256"],
        "maximumParentRegistrationDistanceM": old_max, "tieToleranceM": old_tie,
        "cachedArtifactSHA256": artifacts, "cacheReceipt": receipt,
    }

def validate_loaded_context(args: argparse.Namespace, cache: dict[str, Any]) -> dict[str, Any]:
    surfaces = {
        "O": engine.validate_candidate(Path(args.outer), "O", require_f32=True),
        "I": engine.validate_candidate(Path(args.inner), "I", require_f32=True),
        "D": engine.validate_candidate(Path(args.deep), "D", require_f32=True),
    }
    frames, pose_meta, binding_bodies = engine.load_frames(Path(args.poses))
    if len(binding_bodies) != 86:
        raise CacheAdmissionError("current pose snapshot does not define exactly 86 binding lanes")
    parent, index, material_indices, parent_meta = engine.load_parent_current(Path(args.parent))
    weights86, weight_meta = engine.load_parent_full86_weights(
        Path(args.parent_full86_weights), parent, parent_meta)
    expected = cache["cacheReceipt"]["parentCurrent"]
    for field in ("massSHA256", "nodeSHA256", "sourceSHA256", "tetsSHA256",
                  "labelsSHA256", "nodeCount", "tetCount"):
        require_equal(f"loaded parent {field}", expected[field], parent_meta[field])
    require_equal("loaded pose SHA-256", cache["poseSHA256"], pose_meta["sha256"])
    require_equal("loaded parent full86 SHA-256", cache["parentFull86WeightsSHA256"],
                  weight_meta["sha256"])
    if weights86.shape != (len(parent["points"]), 86):
        raise CacheAdmissionError("loaded parent full86 domain is not node-count by 86")
    return {"surfaces": surfaces, "frames": frames, "poseMeta": pose_meta,
            "bindingBodies": binding_bodies, "parent": parent, "parentIndex": index,
            "materialIndices": material_indices, "parentMeta": parent_meta,
            "parentWeights86": weights86, "parentWeightMeta": weight_meta}

def copy_cached_registration_rows(cache_file: Path, output_file: Path,
                                  expected_counts: dict[str, int]) -> dict[str, int]:
    expected_header = (",".join(engine.PARENT_HEADER) + "\n").encode("utf-8")
    counts = {"O": 0, "I": 0}
    with cache_file.open("rb") as source, output_file.open("wb") as target:
        if source.readline() != expected_header:
            raise CacheAdmissionError("cached registration header differs from frozen 398 schema")
        target.write(expected_header)
        for line in source:
            layer = line.partition(b",")[0]
            if layer == b"O":
                target.write(line); counts["O"] += 1
            elif layer == b"I":
                target.write(line); counts["I"] += 1
    for role in ("O", "I"):
        require_equal(f"cached {role} registration row count",
                      expected_counts[role], counts[role])
    return counts

def append_registration_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=engine.PARENT_HEADER,
                            extrasaction="raise", lineterminator="\n")
    writer.writerows(rows)
    with path.open("ab") as output:
        output.write(buffer.getvalue().encode("utf-8"))

def export_cached(args: argparse.Namespace) -> dict[str, Any]:
    cache = validate_cache_inputs(args)
    context = validate_loaded_context(args, cache)
    if args.out is None:
        raise CacheAdmissionError("cached export requires explicit --out")
    if args.out.exists():
        raise CacheAdmissionError(f"refuse existing output directory {args.out}")
    staged = args.out.with_name(args.out.name + ".staging")
    if staged.exists():
        raise CacheAdmissionError(f"refuse existing staging directory {staged}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staged.mkdir(parents=True)
    try:
        old = cache["cacheReceipt"]
        cache_dir = Path(args.reuse_oi_cache_dir)
        copied = copy_cached_role_artifacts(cache_dir, staged, old)
        shutil.copyfile(cache_dir / "donor-frames.csv", staged / "donor-frames.csv")
        require_equal("copied donor-frames.csv",
                      cache["cachedArtifactSHA256"]["donor-frames.csv"],
                      sha256_file(staged / "donor-frames.csv"))
        expected_counts = {r: int(old["results"][r]["nodes"]) for r in ("O", "I")}
        registration_path = staged / "node-parent-registration.csv"
        registration_counts = copy_cached_registration_rows(
            cache_dir / "node-parent-registration.csv", registration_path, expected_counts)

        surfaces = context["surfaces"]
        d_result = engine.register_layer(
            surfaces["D"], context["parent"], context["parentIndex"],
            context["materialIndices"], context["frames"], context["parentWeights86"],
            context["bindingBodies"], args.maximum_parent_distance_m, args.tie_tolerance_m)
        engine.write_csv(staged / "registered-D.csv", engine.NODE_HEADER, d_result["nodes"])
        engine.write_csv(staged / "registered-D-faces.csv", engine.FACE_HEADER, d_result["faces"])
        engine.write_csv(staged / "parent-tet-full-weights-D.csv",
                         engine.WEIGHT_HEADER, d_result["weightRows"])
        engine.write_csv(staged / "parent-tet-full-weight-owners-D.csv",
                         engine.WEIGHT_OWNER_HEADER, d_result["weightOwnerRows"])
        append_registration_rows(registration_path, d_result["registrations"])

        registry = list(old["sourceDomainRegistry"][:2]) + [d_result["sourceDomain"]]
        (staged / "source-domain-registry.json").write_text(
            json.dumps(registry, indent=2, sort_keys=True) + "\n")
        fallback_reports = {}
        for role in ("O", "I"):
            report = old.get("exactParentNearTieFallbackReports", {}).get(role)
            if report:
                shutil.copyfile(cache_dir / report["path"], staged / report["path"])
                require_equal(f"copied fallback report {role}", report["sha256"],
                              sha256_file(staged / report["path"]))
                fallback_reports[role] = report
        if d_result.get("exactNearTieFallbacks"):
            report_path = staged / "exact-parent-near-tie-fallbacks-D.json"
            report_path.write_text(json.dumps({
                "schema": "exact-parent-near-tie-fallbacks.v1", "layer": "D",
                "candidatePath": surfaces["D"]["path"],
                "candidateSHA256": surfaces["D"]["sha256"],
                "parentCurrentSHA256": context["parentMeta"]["nodeSHA256"],
                "poseSnapshotSHA256": context["poseMeta"]["sha256"],
                "fallbacks": d_result["exactNearTieFallbacks"],
                "physical_admission": False, "native_steps": 0,
            }, indent=2, sort_keys=True) + "\n")
            fallback_reports["D"] = {
                "path": report_path.name, "sha256": sha256_file(report_path),
                "count": len(d_result["exactNearTieFallbacks"])}
        metrics = {"O": old["results"]["O"], "I": old["results"]["I"],
                   "D": d_result["metrics"]}
        candidate_surfaces = {
            "O": old["candidateSurfaces"]["O"], "I": old["candidateSurfaces"]["I"],
            "D": {"path": surfaces["D"]["path"], "sha256": surfaces["D"]["sha256"],
                  "points": surfaces["D"]["vertexCount"], "faces": surfaces["D"]["faceCount"],
                  "sourceOriginalFaceIDs": surfaces["D"]["sourceOriginalFaceIDs"],
                  "sourceFaceAncestry": surfaces["D"]["faceAncestry"]}}
        output_hashes = {p.name: sha256_file(p) for p in staged.iterdir() if p.is_file()}
        receipt = {
            "schema": "registered-authoring-surface-sidecars-receipt.v1",
            "status": "reference_registration_ready_geometry_not_admitted",
            "physical_admission": False, "geometry_admission": False, "native_steps": 0,
            "method": "O/I sidecars are reused byte-for-byte only after exact candidate, parent mass/nodes/source/tets/labels, native pose, full86, producer and cache receipt SHA checks. D is registered afresh by the frozen 398 engine. Geometry admission remains separate.",
            "source_ancestry_claimed_from_D37_for_candidate_nodes": False,
            "face_level_source_ancestry_preserved": True,
            "face_ancestry_source_arrays": "face_source_original_id,face_source_identity0,face_source_identity1 plus pinned face_source_identity_schema_by_original_id_json in each candidate NPZ",
            "candidateSurfaces": candidate_surfaces, "sourceDomainRegistry": registry,
            "parentCurrent": context["parentMeta"], "poseSnapshot": context["poseMeta"],
            "parentFull86Weights": context["parentWeightMeta"],
            "full86WeightMethod": old["full86WeightMethod"],
            "bindingBodyByLane": context["bindingBodies"], "limits": old["limits"],
            "results": metrics, "outputSHA256": output_hashes,
            "exactParentNearTieFallbackReports": fallback_reports,
            "cacheReuse": {
                "schema": "registered-authoring-role-cache-reuse.v1",
                "cacheRoles": ["O", "I"], "recomputedRoles": ["D"],
                "cacheReceiptPath": cache["cacheReceiptPath"],
                "cacheReceiptSHA256": cache["cacheReceiptSHA256"],
                "cacheLaunchPath": cache["cacheLaunchPath"],
                "cacheLaunchSHA256": cache["cacheLaunchSHA256"],
                "frozenExporterPath": cache["enginePath"],
                "frozenExporterSHA256": cache["engineSHA256"],
                "cacheVariantSHA256": cache["cacheVariantSHA256"],
                "verifiedInputs": {
                    "candidateSHA256": cache["candidateSHA256"],
                    "parentSHA256": cache["parentSHA256"],
                    "poseSHA256": cache["poseSHA256"],
                    "parentFull86WeightsSHA256": cache["parentFull86WeightsSHA256"],
                    "maximumParentRegistrationDistanceM": cache["maximumParentRegistrationDistanceM"],
                    "tieToleranceM": cache["tieToleranceM"]},
                "copiedRoleArtifactSHA256": copied,
                "registrationRowsCopied": registration_counts},
        }
        if fallback_reports:
            receipt["exactParentNearTieFallbackPolicy"] = old.get(
                "exactParentNearTieFallbackPolicy",
                "The frozen 398 exact-near-tie fallback policy applies to reused O/I and new D rows.")
        (staged / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, args.out)
        receipt["receiptSHA256"] = sha256_file(args.out / "receipt.json")
        print(json.dumps({"output": str(args.out), "receiptSHA256": receipt["receiptSHA256"],
                          "results": metrics, "cacheReuse": receipt["cacheReuse"],
                          "physical_admission": False, "geometry_admission": False,
                          "native_steps": 0}, indent=2))
        return receipt
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise

def cache_preflight(args: argparse.Namespace) -> dict[str, Any]:
    cache = validate_cache_inputs(args)
    context = validate_loaded_context(args, cache)
    return {
        "schema": "registered-authoring-role-cache-preflight.v1",
        "status": "hash_and_schema_preflight_passed_no_registration",
        "cacheReceiptSHA256": cache["cacheReceiptSHA256"],
        "cacheLaunchSHA256": cache["cacheLaunchSHA256"],
        "frozenExporterSHA256": cache["engineSHA256"],
        "cacheVariantSHA256": cache["cacheVariantSHA256"],
        "candidateSHA256": cache["candidateSHA256"],
        "parentSHA256": cache["parentSHA256"],
        "poseSHA256": cache["poseSHA256"],
        "parentFull86WeightsSHA256": cache["parentFull86WeightsSHA256"],
        "cachedArtifactSHA256": cache["cachedArtifactSHA256"],
        "candidateCounts": {r: {"nodes":context["surfaces"][r]["vertexCount"],
                                "faces":context["surfaces"][r]["faceCount"]}
                            for r in ("O", "I", "D")},
        "parentNodeCount": context["parentMeta"]["nodeCount"],
        "parentTetCount": context["parentMeta"]["tetCount"],
        "bindingCount": len(context["bindingBodies"]),
        "registrationCalls": 0, "runtime_created": 0, "accepted_steps": 0,
        "physical_admission": False, "geometry_admission": False}

def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    for name in ("outer", "inner", "deep", "parent", "poses", "parent-full86-weights", "out"):
        ap.add_argument("--" + name, type=Path)
    ap.add_argument("--maximum-parent-distance-m", type=float, default=0.02)
    ap.add_argument("--tie-tolerance-m", type=float, default=1.0e-8)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--sample-check-count", type=int, default=0)
    ap.add_argument("--reuse-oi-cache-dir", type=Path)
    ap.add_argument("--reuse-oi-cache-launch", type=Path)
    ap.add_argument("--reuse-oi-cache-receipt-sha256")
    ap.add_argument("--reuse-oi-cache-launch-sha256")
    ap.add_argument("--reuse-oi-engine-sha256")
    ap.add_argument("--cache-preflight-only", action="store_true")
    return ap

def main() -> None:
    args = make_parser().parse_args()
    cached = any((args.reuse_oi_cache_dir, args.reuse_oi_cache_launch,
                  args.reuse_oi_cache_receipt_sha256, args.reuse_oi_cache_launch_sha256,
                  args.reuse_oi_engine_sha256, args.cache_preflight_only))
    if not cached:
        engine.main()
        return
    if args.preflight_only or args.sample_check_count:
        raise CacheAdmissionError("cached O/I reuse cannot be combined with baseline preflight/sample modes")
    if args.cache_preflight_only:
        print(json.dumps(cache_preflight(args), indent=2, sort_keys=True))
        return
    export_cached(args)

if __name__ == "__main__":
    main()
