"""Bind explicitly repaired passive rows using the existing NHTISS and anatomy receipt owners."""
from pathlib import Path
import json
from . import passive_attachment_composition as pac
from .passive_reference_binding_extension import extend_reference_bindings


def _validate_parent_vertex_fields(before: dict, after: dict,
                                   stable_ids: tuple[int, ...]) -> None:
    """Allow index edits only when every existing material allocation is exact."""
    old = {int(row[6]): row for row in before["records"]}
    new = {int(row[6]): row for row in after["records"]}
    for sid in stable_ids:
        pac.require(sid in old and sid in new,
                    "parent-mapped row identity is missing")
        a, b = pac._row_slices(before, old[sid]), pac._row_slices(after, new[sid])
        pac.require(all(a[key] == b[key] for key in
                        ("vertex_bytes", "binding_bytes", "vertex_count",
                         "binding_count", "stable_id", "layer")),
                    "changed parent-mapped row requires material-parent reconstruction")


def compose_reference_receipt(source_receipt: Path, output: Path,
                              replacements: list[tuple[int, Path, Path]], *,
                              extensions: dict[int, int]) -> dict:
    """Compose explicit passive rows and retain all untouched parent-map owners."""
    import subprocess

    source_receipt, output = Path(source_receipt).resolve(), Path(output).resolve()
    receipt = json.loads(source_receipt.read_text())
    pac.require(receipt.get("schema") == "numi.human.resting-anatomy-receipt.v1",
                "reference receipt schema")
    owner = receipt["provenance"]["native_muscle_surfaces"]
    old = Path(owner["payload_path"])
    if not old.is_absolute():
        old = source_receipt.parent/old
    pac.require(pac._surface_sha256(old) == owner["sha256"], "prior muscle owner differs")
    changed = tuple(sorted(sid for sid, _, _ in replacements))
    pac.require(len(set(changed)) == len(changed) and set(extensions) <= set(changed),
                "binding extension must belong to an explicit replaced row")
    parents = receipt.get("passive_tissue_parent_bindings")
    if parents is not None:
        pac.require(parents["child_payload_sha256"] == owner["sha256"],
                    "prior parent-map child identity differs")
        mapped = {r["stable_id"] for r in parents["surfaces"]}
    else:
        mapped = set()
    pac.require(not output.exists(), "reference receipt output exists")
    output.mkdir()
    if extensions:
        extension = extend_reference_bindings(old, output/"supports", extensions)
        extended = output/"supports"/old.name
    else:
        # Existing supports already describe both sides of a prior interface.
        # Reuse them without constructing a duplicate payload or binding table.
        extension = {"status": "EXISTING_REGISTERED_SUPPORTS_REUSED",
                     "source_path": str(old), "source_sha256": owner["sha256"],
                     "changed_binding_rows": []}
        extended = old
    composed = pac.compose(extended, output/"composed", replacements,
                           reference_surface_rows=changed)
    payload = output/"composed"/old.name
    before, after = pac._read_nhtiss4(old), pac._read_nhtiss4(payload)
    pac.require(before["surface_count"] == after["surface_count"]
                and before["source_digest"] == after["source_digest"]
                and before["fingerprint"] == after["fingerprint"],
                "reference composition changed source or registration")
    # An index-only repair retains every material vertex and its exact field.
    # Positions, normals, binding indices, weights and body frames must all
    # match; allocation or field edits still require parent reconstruction.
    index_only_mapped = tuple(sorted(mapped.intersection(changed)))
    _validate_parent_vertex_fields(before, after, index_only_mapped)
    unchanged = []
    for a, b in zip(before["records"], after["records"]):
        pac.require(int(a[6]) == int(b[6]) and int(a[7]) == int(b[7]),
                    "reference composition changed typed inventory")
        if int(a[6]) in changed:
            continue
        x, y = pac._row_slices(before, a), pac._row_slices(after, b)
        pac.require(all(x[k] == y[k] for k in ("vertex_bytes", "binding_bytes", "local_faces")),
                    "reference composition changed an unselected row")
        unchanged.append(int(a[6]))
    manifest_path = payload.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    metadata = manifest["payload"]
    previous = dict(owner)
    owner.update(payload_path=str(payload), sha256=metadata["sha256"],
                 manifest_path=str(manifest_path), manifest_sha256=pac._surface_sha256(manifest_path),
                 surface_count=metadata["surface_count"], body_binding_count=metadata["binding_count"],
                 vertex_count=metadata["vertex_count"], index_count=metadata["index_count"],
                 registration_fingerprint32=metadata["registration_fingerprint32"])
    if parents is not None:
        # All parent-mapped local rows were independently compared above.
        # Global allocation offsets may move; the runtime joins typed row IDs.
        parents["child_payload_sha256"] = owner["sha256"]
    common = receipt["provenance"].get("cardiac_geometry_binding", {}).get("common_field")
    if common:
        for key in ("map", "polynomials", "domain_boxes"):
            record = common[key]
            asset = Path(record["path"])
            if not asset.is_absolute():
                asset = source_receipt.parent/asset
            pac.require(pac._surface_sha256(asset) == record["sha256"], "cardiac source differs")
            target = output/asset.name
            pac.require(not target.exists(), "cardiac destination collides")
            subprocess.run(["/bin/cp", "-c", "-p", str(asset), str(target)], check=True)
            pac.require(pac._surface_sha256(target) == record["sha256"], "cardiac copy differs")
            record["path"] = target.name
    anatomy = Path(receipt["payload"]["path"])
    if not anatomy.is_absolute():
        anatomy = source_receipt.parent/anatomy
    pac.require(pac._surface_sha256(anatomy) == receipt["payload"]["sha256"],
                "anatomy source differs")
    receipt["payload"]["path"] = str(anatomy.resolve())
    proof = {"prior_receipt_path": str(source_receipt),
             "prior_receipt_sha256": pac._surface_sha256(source_receipt),
             "prior_muscle_owner": previous,
             "changed_stable_ids": list(changed),
             "unchanged_row_local_vertex_binding_and_face_bytes": unchanged,
             "binding_extension": extension, "geometry_composition": composed,
             "parent_mapped_rows_source_carriers_and_terms_unchanged": True,
             "parent_mapped_index_only_rows": list(index_only_mapped),
             "physical_routes_mass_tendon_activation_and_cardiac_state_unchanged": True,
             "native_admission": False, "anatomical_admission": False}
    receipt["provenance"].setdefault("passive_reference_receipt_compositions", []).append(proof)
    result = output/"resting-anatomy-receipt.json"
    result.write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    (output/"report.json").write_text(json.dumps(proof, indent=2, sort_keys=True)+"\n")
    return {"receipt_path": str(result), "receipt_sha256": pac._surface_sha256(result),
            "payload_path": str(payload), "payload_sha256": pac._surface_sha256(payload)}


def main(argv=None):
    """Pin local inputs before invoking the existing passive composition owner.

    This command prepares anatomy sources. It neither runs geometry predicates
    nor grants motion/native admission; the shared resting verifier owns those
    checks.
    """
    import argparse
    import re

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replacement", nargs=5, action="append", required=True,
                        metavar=("ID", "NPZ", "NPZ_SHA256", "REPORT", "REPORT_SHA256"))
    parser.add_argument("--extend", nargs=2, type=int, action="append", default=[],
                        metavar=("TARGET_ID", "SUPPORT_SOURCE_ID"))
    parser.add_argument("--source-pin", nargs=2, action="append", default=[],
                        metavar=("PATH", "SHA256"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    pins = {}
    def pin(path, digest):
        path = Path(path).resolve()
        pac.require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                    "reference composition SHA256 must be lowercase hex")
        pac.require(path not in pins or pins[path] == digest,
                    "conflicting reference composition source pins")
        pac.require(path.is_file() and pac._surface_sha256(path) == digest,
                    f"reference composition source hash differs: {path}")
        pins[path] = digest
        return path

    receipt = pin(args.receipt, args.receipt_sha256)
    replacements = []
    for stable_id, npz, npz_sha, report, report_sha in args.replacement:
        pac.require(stable_id.isascii() and stable_id.isdecimal()
                    and int(stable_id) > 0, "invalid passive stable ID")
        replacements.append((int(stable_id), pin(npz, npz_sha), pin(report, report_sha)))
    changed = [row[0] for row in replacements]
    pac.require(len(set(changed)) == len(changed), "duplicate passive replacement ID")
    extensions = {}
    for target, source in args.extend:
        pac.require(target in changed and source > 0 and target not in extensions,
                    "invalid or duplicate passive binding extension")
        extensions[target] = source
    for path, digest in args.source_pin:
        pin(path, digest)
    pac.require(not args.output.exists(), "reference receipt output exists")
    if args.preflight_only:
        print(json.dumps({"status": "PASS_PINNED_REFERENCE_COMPOSITION_INPUTS_ONLY",
                          "changed_stable_ids": changed,
                          "geometry_predicates_executed": False,
                          "native_admission": False}))
        return 0
    result = compose_reference_receipt(receipt, args.output, replacements,
                                       extensions=extensions)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
