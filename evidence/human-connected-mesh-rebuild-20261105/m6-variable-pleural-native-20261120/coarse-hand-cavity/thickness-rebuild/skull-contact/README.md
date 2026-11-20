# Repair skull-shell crossings with exact trial checks

This SSH Numi turn starts from the exact native initial geometry and the
same interior-edge topology as 73c3a63. The source shell, skull banks,
runtime binary, bed rows, anatomical labels, and rest mass ledgers are all
pinned by the preceding work. These tests author candidate meshes offline.

The new fit includes finite-triangle separating planes from actual skull
crossings, with larger penalties on deep faces. Pair identity uses the
original shell face and the individual native bone triangle. Exact Float32
queries decide each line search. During local restoration, only columns
touched by failed cell/solid/width checks return to their previously accepted
authoring positions. The full trial is then checked again. This prevents
one near-degenerate cell from forcing every column in the batch to take
the same small step. It creates no runtime forces or locks.

| Candidate | Outer skull | Interface skull | Deep skull | Whole shell against bone |
|---|---:|---:|---:|---:|
| Matched native source | 702 | 2,311 | 2,685 | 17,878 |
| Previous interior-edge fit | 500 | 1,526 | 3,100 | 17,306 |
| Triangle-only guard with local restoration | 424 | 944 | 2,471 | 16,019 |
| Add closed-solid and width guards | 508 | 1,350 | 2,539 | 16,577 |
| Continue guarded candidate | 506 | 1,350 | 2,533 | 16,569 |

The triangle-only fit introduced 41 inside pairs in the changed-vertex
audit and made some columns wider than their inferred reference limits.
The guarded continuation checks all 8,343 shell-boundary vertices
against the eight separate closed skull meshes using the exact indexed
point locator. With unchanged topology, it reduces 169 inside pairs to 96
and adds none. The initial 169 refer to the refined audit vertices and
individual bone volumes, including all shell layers. The original native
audit sampled coarse outer/deep vertices by body bank. There are zero
boundary or indeterminate points in the new check. Frame emission stays
inside the paired-layer width limits and their 0.2 μm rounding allowance.

Each retained tet stays positive in both frames. The independent verifier
closes the total inferred 13.93968 kg skin/fat mass, original typed source
quotas, destination masses and the distance-bound routes. Bed vertices
are unchanged. No new self-intersection or external bone-bank pairs were
found. The remaining count includes 1,587 new pair identities within the
same skull bank. Both the triangle and solid errors remain nonzero.

The guarded fit takes about 90 s offline. The 141 s warm continuation clears
only eight further pairs. Its trace repeatedly rejects larger candidates
on determinant failures and smaller candidates on deep contact regression.
The most frequent determinant/shape witnesses include original cells 13979,
13925, 12614 and 12824. The retained witness positions and ancestry are
available for a connected local cavity rebuild. Repeating this fit without
changing the mesh geometry has diminishing returns.

The first direct-plane fit had zero admissible trial moves under the cell
checks. The first restored trial falsely read an exhausted triangle iterator
as a cleared bank and is excluded. The test suite now repeats a crossing
query after a disjoint trial and checks its identities and counts. It also
checks the vectorized plane choice against the scalar owner, every affine
corner inequality, and new exact boundary or containment outcomes.

There is no native motion qualification for these candidates. The retained
coupled scene admitted the 34 registered ligament rows during its short
release test. The physical bed supported 0.40 N of about 953 N weight.
Supported shoulder loading and equilibrium remain unqualified. Its
measured variable-pressure body-only prefix ran 2 ms at 0.00002028×.
The zero-crossing 16 s gate, continuous variable-pressure motion, and
performance targets remain open.

## Reproduction

On SSH Numi, create `skull-contact-036` alongside `edge-cavity-035`.
Restore the four Python files from the interior-edge experiment and apply
`layer-fit.patch` with `patch -p0` inside the new directory. Add the
other Python files from this evidence directory. The rest ledger and
binary inputs follow the preceding recovery receipts.

```sh
python3 test_skull_contact.py
python3 rebuild.py guarded --whole-skull --all-edges --interior-edge 3263 5829 --depth-contact --restore-local --preserve-solid-and-width
python3 rebuild.py continuation --whole-skull --all-edges --interior-edge 3263 5829 --depth-contact --restore-local --preserve-solid-and-width --resume guarded
python3 verify_rebuild.py guarded continuation
python3 audit_solid_vertices.py continuation --all-surface
```

Raw candidates, exact line-search witnesses, and source-dependent audit
tables were restored from the private archive and checked against the
manifest. Seven generated publication indexes were removed after recreating
and byte-comparing their staged listings with commits reachable on main.
