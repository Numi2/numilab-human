# Rodero case 18: coordinate-preserving topology normalization is insufficient

The source cannot yield fully embedded ventricular cavity boundaries merely by
duplicating coincident node identities while retaining every original tetrahedral
face connection. A maximal such split resolves 29 of 31 abstract boundary edge
defects, but two remain. Even the RV candidate, whose abstract boundary becomes
an oriented sphere, retains exact geometric self-contact. This is a bounded
source audit; no source mesh, production code or physical state was changed.

## What was evaluated

The physical Mac mini read the existing 300,965-node, 1,470,083-tetrahedron asset
at `/Users/n/human-cardiac-wall-anatomy-20260912/asset`. Every input buffer matched
its manifest SHA256. Human owner revision was `ae6aaa97afd6c139969025a83aa04644856667e5`;
native revision was `f73b07133d09ceb9ed7ff94d48d5863157f2527b`, clean when checked.
No native/GPU/build workload was running before this bounded CPU audit.

For each of the 47 defective boundary vertices, incident tetrahedra were joined
only across their actual shared triangular faces containing that vertex. Every
such face requires the same nodal identity on both sides. The connected
components therefore give the finest possible vertex duplication that preserves
all existing shared tetrahedral faces. Each cell retains its four coordinates,
label, frame and exact determinant; every original boundary face remains present.

All 31 defective edges have two disconnected, path-shaped tetrahedral links:
two material fans, with four boundary faces. An exact integer determinant test
found no positive-angle overlap among their incident tetrahedral sectors. This
local result does not qualify global tetrahedral embeddedness.

Nineteen vertices admit two distinct star components. The scratch mapping adds
19 coincident node identities; it changes no coordinate or cell volume. An
independent pass checked all 993 affected shared-face constraints. It does release
the original shared-node displacement constraints between the separated fans;
it is not mechanically equivalent to the original nodal FEM discretization.

| Boundary component | Original edge / vertex defects | After maximal split | Result |
| --- | ---: | ---: | --- |
| 0, LV candidate | 4 / 4 | 1 / 2 | Still nonmanifold; Euler 3 |
| 1, LA | 0 / 0 | 0 / 0 | Unchanged |
| 2, outer boundary | 25 / 40 | 1 / 2 | Still nonmanifold; Euler −1 |
| 3, RV candidate under source valve interpretation | 2 / 3 | 0 / 0 | Abstract sphere; geometric self-contact remains |
| 4, RA | 0 / 0 | 0 / 0 | Unchanged |
| 5, tiny unassigned void | 0 / 0 | 0 / 0 | Unchanged |

All six boundary components retain precisely their original face sets and signed
volume integrals. The RV candidate still contains 42,506 faces and encloses the
same signed integral, −119.1683098162639 mL in material-outward orientation.
This is not a biological volume qualification.

## Two exact continuity obstructions

The remaining LV edge is `[194623, 222142]`. Its two fans contain source cells
`[679124,1032607]` and `[231210,1039476]`, all label 1. Both edge endpoints have
one face-connected tetrahedral star. For example, preserving faces through
vertex 194623 forces equal nodal identity along the cell path
`679124 → 1032607 → 347955 → 1270165 → 715051 → 1023174 → 1036422 → 231210`.
The independent path through vertex 222142 is also retained in `report.json`.
Thus neither endpoint can separate those fans without breaking an existing
shared tissue face.

The remaining outer edge is `[131663,280055]`. One fan contains cells
`[896330,1089313]` with source pulmonary-valve label 10; the other is cell
`462514`, pulmonary-artery label 6. Both endpoint stars again contain explicit
shared-face paths connecting the fans. `edge_link_reports` records every path,
face, cell, label and link graph, including these two impossibility witnesses.

This proves the limitation for ordinary conforming nodal tetrahedral FEM under
fixed cell coordinates/connectivity and preserved shared-face continuity. It
does not rule out a separately specified geometric reconstruction, a new contact
or crack model, or selection of a different source.

## Why the abstract RV repair is not an embedded cavity

Splitting source vertex 170947 creates coincident node 300972 solely for RV cell
1309400; the other twelve incident cells, label 1, keep node 170947. This resolves
the two RV edge incidences combinatorially. However, faces 29270 and 29271 now
share only node 26256 while still intersecting geometrically along the entire
segment from node 26256 to the coincident 170947/300972 position. The exact
intersection owner rejects that contact beyond their shared topological vertex.

Across the 29 abstractly resolved edges, the bounded exact check finds 116
forbidden face-contact pairs: 12 LV, 96 outer and 8 RV. Every pair and its exact
rational intersection coordinates is in `split-contacts.json`. This is sufficient
to reject embeddedness; it is not an exhaustive inventory of all global contacts.
Admitting this variant as a clean cavity would hide a source defect. No cap was
added, no geometric gap was introduced, and no intersection tolerance was used.

## RV tissue-volume discrepancy

Independent exact determinant sums reproduce the imported label volumes:

| Observable | Source tetrahedra | Published case-18 CSV | Difference |
| --- | ---: | ---: | ---: |
| LV label 1 | 87.68573554497355 mL | 87.6857353647 mL | +0.00000018027356 mL |
| RV label 2 | 45.024903409970136 mL | 47.4406584127 mL | −2.415755002729867 mL (−5.092%) |

Adding only the adjacent tricuspid layer (label 8) gives 47.02016085117169 mL;
adding only the pulmonary layer (10) gives 46.081927067478155 mL; adding both gives
48.07718450867971 mL. None reproduces the CSV. These layers are artificial source
valve tissue, not an authorized reassignment of RV myocardium.

The nodal UVC `V` field also provides no source-supported missing RV volume.
All 722,773 LV cells have `V=-1` at every node. RV cells comprise 362,459 all-`+1`
cells (43.68386999848 mL), 12,214 mixed `−1/+1` cells (1.33565969622 mL), and 88
all-`−1` cells (0.00537371527 mL). There is no label-1 cell with a supplied `+1`
node to justify moving LV tissue into RV by that field. UVC values on neighboring
nonventricular labels include the source `−10` sentinel; they are not a replacement
for regional material labels.

S1 section S1.1 specifies a constructed 3.5 mm RV wall thickness, and Fig B labels
RV myocardium as 2. Section S1.2 describes source surface extraction but gives no
different volume membership rule explaining this CSV mismatch. The single-file
archive supplies neither an alternate case-18 mesh nor the original simulation
input deck. The discrepancy remains unresolved. Pure node duplication cannot
alter it because all per-label determinants are invariant. The near-exact LV
agreement also rules out a common global unit conversion as the explanation.

## Decision and evidence boundary

Do not promote the scratch normalization to anatomical FEM/cavity admission.
Retain the source as imported and the defects as explicit failed gates. A
ventricular path requires source-authorized geometry reconstruction or another
source whose tissue and lumen boundaries pass the permanent embedding checks.
RA/LA retain their previously qualified local geometric correspondence; that does
not establish full-wall embeddedness, material calibration, artificial-cap port
ownership or blood/tissue mass partition.

`execution.json` binds commands, scripts, successful reports, the source/owner
pins and one retained contact-script syntax failure before correction. The star,
topology, exact volume and UVC pass took 6.2395 s; the exact contact-counterexample
pass took 0.2364 s. No physical stepping, native build or GPU run occurred.

Primary pins:

- Source archive: `b50919a711dc3914cc0a4dd9cabc19679a9f36be9ac6eb26a23ca6799f501720`.
- Source VTK member: `f0d4f3fc21ea229fa1b334888d374ee606a2915b414fff047764aad44f1c0a41`.
- Asset manifest: `7eb93af573b328c77f3960f3f9fb40e337356d64fa6fa3ff59e258d1959b0dee`.
- Source CSV: `3f59ea2e6388aa370efe9046ecd5b4fd8bee87037aa904f45a80356f3932edcd`.
- S1 PDF: `08777693dcfd5cfaac676c3e015761d62b68706af21577e678f89808a0c979ba`.
- Exact intersection owner: `8b138882161c78312cb9b5aa46f49795092eb9b67ed9f705dc354787cb0a635b`.
