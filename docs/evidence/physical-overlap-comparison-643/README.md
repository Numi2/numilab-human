# Complete coarse tissue/bone overlap comparison

The connected O/I/D candidate `fca6d6e7…32778b` preserves all 29,642 positive emitted cells and the fixed 534 child-mass/density constraints, but it does not improve physical clearance. Across all 185 matched bank operands, skin intersection decreases in 7 banks, while fat intersection increases in 5 banks and decreases in 3. Both original006 and the fit retain 112 banks with positive material intersection. Per-bank intersection volumes overlap; they are not a union volume.

The reusable comparator checks the complete matched bank identities, material presence, Boolean status, finite volumes, provenance and coordinate domain. It reports each signed change and returns exit 2 if any material/bank intersection increases. It applies no positive-volume tolerance. Its result is a diagnostic; even a nonincreasing comparison never grants physical admission.

Six focused tests pass, including a skin improvement that hides a fat regression, tiny positive intersections, incomplete/duplicate/mismatched operands, invalid Boolean results, and an improvement that remains diagnostic. The actual candidate comparison returns the expected exit 2.

Inputs are the retained receipts from the existing `audit_coarse_material_bone_volume_427.py` owner. Each receipt pins its candidate NPZ, all185 bank manifest, script hash and Float32/reference coordinate domain. Existing source assets and current geometry remain untouched. No Runtime was created, no physical step was accepted, and native motion remains unverified.

Run:
```sh
python3 -m unittest -v test_coarse_material_bone_overlap_641
python3 compare_coarse_material_bone_overlap_641.py --baseline original006-427.json --candidate fit040-004-427.json --out fresh-comparison.json
```
