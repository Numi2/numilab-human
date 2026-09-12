# BodyParts3D atrial wall–cavity admission audit

Neither pinned atrial wall can directly supply the material boundary of its existing cavity reference. Both wall solids are individually embedded, but the right atrial cavity crosses its wall in **3,000 triangle pairs**, and the left in **1,880**. Each cavity has exact face-centroid witnesses both inside and outside the wall solid. The pairs have **zero shared vertices and zero shared triangles**. These are mismatched independently authored surfaces, not conforming wall/lumen interfaces.

## Source and executed geometry

The audited archive is `Sources/partof_BP3D_4.0_obj_99.zip`, 64,888,505 bytes, SHA256 `9fbc713fffeee924a5a657d9813d84d7eb957bded63adb854931dd5e3eb61c97`. Its bytes and both anatomy-table hashes were verified against `sources.lock.json` SHA256 `03d20d46bfb47b67fb251182758eebbc8f1bbe2f38ce51881dc4c2694774fdee`. Both `partof_element_parts.txt` and `isa_element_parts.txt` identify FJ2439 exclusively as FMA9457, wall of right atrium, and FJ2438 as FMA9531, wall of left atrium.

| Surface | Member SHA256 | Quotient vertices / faces | Raw seam edges → quotient | Euler / genus | Exact oriented volume, mL |
|---|---|---:|---:|---:|---:|
| Right atrial wall FJ2439 | `e7c21cd1659eced056eb56e28f4fa9019ace451ea6e0333097846177f807bb5a` | 8,890 / 17,784 | 1,826 → 0 | −2 / 2 | 27.625785019521302 |
| Left atrial wall FJ2438 | `187235f3c3612ef27abde924d01c71579c2946435e64319164f43e5ad008284d` | 3,766 / 7,544 | 1,102 → 0 | −6 / 4 | 40.53763785499966 |
| Right atrial cavity FJ2424 | `8bd340059ff697cfe9ce57f2427e8c5a982e248a1d2230d477a4fe094adc2a2a` | 1,124 / 2,244 | 146 → 0 | 2 / 0 | 84.5515320178338 |
| Left atrial cavity FJ2425 | `223233fef054fb67b81d27881f10a1acce47c9ade147546b6859fa90e4e37dc3` | 534 / 1,064 | 126 → 0 | 2 / 0 | 51.93679901140525 |

Only exactly equal authored decimal coordinates were identified: 997 right-wall and 518 left-wall duplicate vertices. Every resulting surface is connected, closed, consistently oriented and vertex-manifold, with no degenerate triangles or forbidden self-intersections. Zero tolerance, no vertex movement, no added faces, no source smoothing and no physical stepping were used.

The intersection predicates operate on exact rational values of the published Float64 metre coordinates. Signed zeroth, first and raw second moments use the existing exact rational moment owner. Header volumes, 27.661700 and 40.493400 cm³, differ slightly from the actual admitted mesh integrals and were not substituted for them. Mesh volume is geometry evidence, not blood or tissue mass and not biological calibration.

The source OBJ headers retain the original CC-BY-SA-2.1-Japan notice; the current source lock separately records the database CC-BY-4.0 license. Full headers are preserved in the machine-readable results. This audit does not resolve that provenance distinction into a new licensing claim.

## Concrete overlap witnesses

Right cavity face 0 has a centroid outside the right wall; face 1120 has one inside it. Left cavity face 0 has a centroid inside the left wall; face 66 has one outside it. Existing exact ray parity certifies each witness. Exact rational centroid coordinates, source vertex indices, ray and crossing counts are in `summary.json`; all intersecting source face pairs are retained in `results/atrial-wall-audit.json`.

These witnesses establish actual domain overlap and boundary mismatch. The reported pair counts include every triangle contact under the existing exact predicate; they are not estimates of overlap volume. No overlap volume or repaired surface is claimed.

## Construction consequence

The original wall solids are candidates for a constrained tissue-volume tetrahedralization, subject to exact boundary preservation, positive element quality, source scale, and independent material/density admission. That construction alone does not provide the existing cavity with a complete pressure boundary. The walls have one connected boundary with handles, not separate inner and outer closed boundary components enclosing the cavity reference.

Subtracting the cavity from the wall would change source tissue geometry while leaving parts of the cavity boundary exposed to no material wall. Filling those parts, offsetting or thickening a surface, and treating geometric caps as valves would invent missing anatomy. None is admitted by this audit. An exact Boolean composition would require a separately named model and conservative volume/moment accounting; it would still need physical ownership for every nonmaterial port surface.

The smallest useful next construction is an independently sourced conforming cardiac volume mesh with material region tags, endocardial boundary identity and explicit basal/vascular port semantics. The wall’s endocardial boundary must own the lumen interface. A port closure must retain its actual role and source or modelling provenance; the current native `materialWall` contract must not accept it by relabeling. The BP3D surfaces remain untouched source references for comparison.

## Reproduction

The bounded CPU audit ran on `ssh macmini` after a process check showed no active matching GPU or build jobs. It completed in **13.77 s**, return code 0, using the isolated source snapshot `/Users/n/human-cardiac-partition-20260912/qualification-001`. Its two geometry-owner modules and source lock are byte-identical to Human commit `c4660d4eb33a6aa6bc35cbafb576e2ff7d8d0107`.

```sh
/Users/n/human-cardiac-partition-20260912/venv/bin/python \
  /Users/n/human-cardiac-wall-anatomy-20260912/atrial-audit/audit_atrial_walls.py \
  --root /Users/n/human-cardiac-partition-20260912/qualification-001 \
  --output /Users/n/human-cardiac-wall-anatomy-20260912/atrial-audit/results
```

The offline helper uses an integer AABB tree solely to exclude provably disjoint triangle boxes, then calls the existing Human exact intersection predicates. A deterministic 256-face subset of each surface independently matches the existing direct owner’s full candidate counts and intersection decisions. `summary.json` binds the script, log, full result, source lock and geometry/moment owner hashes. This is source geometry admission evidence only; no native simulation, mass extraction, material calibration or anatomical mechanics is qualified.
