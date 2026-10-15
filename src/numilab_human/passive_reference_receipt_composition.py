"""Bind explicitly repaired passive rows using the existing NHTISS and anatomy receipt owners."""
from pathlib import Path
import json
from . import passive_attachment_composition as pac
from .passive_reference_binding_extension import extend_reference_bindings


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
        pac.require(not mapped.intersection(changed),
                    "changed parent-mapped row requires material-parent reconstruction")
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
             "physical_routes_mass_tendon_activation_and_cardiac_state_unchanged": True,
             "native_admission": False, "anatomical_admission": False}
    receipt["provenance"].setdefault("passive_reference_receipt_compositions", []).append(proof)
    result = output/"resting-anatomy-receipt.json"
    result.write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    (output/"report.json").write_text(json.dumps(proof, indent=2, sort_keys=True)+"\n")
    return {"receipt_path": str(result), "receipt_sha256": pac._surface_sha256(result),
            "payload_path": str(payload), "payload_sha256": pac._surface_sha256(payload)}
