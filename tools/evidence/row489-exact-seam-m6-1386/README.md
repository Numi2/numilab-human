# Exact ulnar-vein seam in the retained native anatomy

The native ulnar-vein surface (stable489, left forearm body92) preserves two vertex slots with exactly equal Float32 coordinates. A self-intersection audit using slot identity alone reports nine contacts at this seam. The exact-coordinate quotient rejoins only coincident coordinates for the predicate; the native geometry, original 2,962 slots and 5,918 face IDs remain unchanged. Every intersection point of each of those nine pairs lies on its shared vertex or edge. Genuine intersections outside a shared simplex remain failures.

## Executed result

The bounded CPU audit ran on SSH Numi, Apple M6, against all eight retained accepted geometry captures from liver-island-native-027-002: steps 0, 3807, 4767, 5023, 5599, 6207, 7423 and 8000. Each capture has nine fully classified seam contacts, zero nonadjacent intersections after the exact quotient and zero degenerate faces. Original face order, duplicate-slot equality, accepted step and exact native Float32 timestep were checked. All input hashes matched again at completion and before publication.

The report retains the exact N27 anatomy, anatomy receipt, run declaration, execution, invocation, accepted packs and receipts, runner, frozen reader and two predicate-owner modules. The latter modules match this repository's sources byte for byte. The runner uses the recorded retained paths and refuses an existing output directory. The copied reader is the exact executed input; its unrelated branch-field main entrypoint is not invoked by this seam audit.

Python compilation and Ruff F821 checks are recorded alongside the copied runner and reader. This is an offline geometric check of actual native captures; it does not run an alternate per-step physics update.

## Preserved failures and limits

The initial reader assumed 524 anatomy records and rejected N27's valid 523-record payload. Its failure remains recorded; the successor validates ABI/version and uses the payload's declared record count. Historical N27 attempt002 had a stale interpretation string naming D12 despite pinning and reading N27. Its unchanged report remains here. Attempt003 corrects that label and adds explicit failing geometry gates.

This finding classifies a preserved source seam; it is not a repair of artery/vein crossings or proof of connected vascular lumens, calibre, vessel-to-muscle clearance or full-body anatomy completion. Those repairs remain pending their separate exact checks and native admission. The rejected forearm fits remain excluded from the working scene, and the working back412 repair is preserved.

The post-restart cleanup summary records reversible compression of superseded artifacts and inactive-cache removal. Current inputs and accepted captures used by this audit remain present and verified. Restart-related swap release is accounted separately from cleanup.
