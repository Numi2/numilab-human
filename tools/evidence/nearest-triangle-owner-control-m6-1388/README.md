# Bounded nearest triangles for anatomical source preparation

Existing source preparation repeatedly scanned approximately 99,000 to 109,000 target faces for nearest-triangle constraints. The bounded index caches a fixed target's centroids and enclosing radii, then delegates exact squared distances and tie-breaking to the existing `myosim_bone_proximity` owner. Targets with at most 1,024 faces keep the full scan because the measured small-target control was slower with an index.

The initial exact scan supplies an upper distance bound. A triangle that could beat it must have a centroid within that distance plus its enclosing radius. All surviving global face IDs, including the initial faces, are sorted before the exact owner scan. The numerical padding protects boundary ties. Callers must keep indexed target geometry fixed for the index lifetime.

On SSH numi M6, the retained benchmark compared 320 deterministic queries over five actual targets in an accepted native geometry capture. Face IDs and Float64 squared distances matched the full owner scan exactly for every query, including sampled vertices and edge midpoints. The intercostal and whole-body skin query loops improved by approximately 66 and 82 times; index builds took approximately 0.015 and 0.017 seconds. These are nearest-query timings, not simulation or full-preparation speedups. The report retains all timings, target semantic/ID pairs, candidate counts and source hashes.

The original slower small-target control is retained. The current helper is used in isolated source preparation; exact intersection, source identity, registered-rest and native accepted-state gates remain mandatory. No anatomical candidate or physiological qualification is admitted by this control.

The retained benchmark's absolute input paths and original output assertion are intentional execution evidence. A new run requires a separately named output. See `retention.json` for exact source/publication identities.

An additional 126-query control passed against the same exact owner. It exercises a long triangle that a nearest-centroid-only shortlist misses, coincident-face global ties, random queries and the small-target fallback. Run the publication control on Numi with its pinned owner checkout:

    /Users/n/numi-human-prep-venv-20261005/bin/python regression.py --owner-source /Users/n/numi-human-left-pq-publication-1344/src --output /absolute/new/report.json
