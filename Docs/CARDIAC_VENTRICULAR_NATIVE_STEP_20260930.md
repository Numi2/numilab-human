# One accepted source-ventricular FEM step — 30 September 2026

The Rodero case18 **ventricular subset** now enters Matter's owning native FEM
transaction. The Apple M4 cooked all **1,097,534 label-1/2 tetrahedra** and
**218,077 shared source nodes** as one non-mixed FEM object, using the source
ventricular Guccione passive law, all corresponding source fibre/sheet-derived
material frames, and the [100 ms candidate active-tension field](CARDIAC_ACTIVE_TENSION_INGRESS_20260930.md).
The native `cookFEMActiveTensions` check proved that every Float32 tension maps
bitwise to the compiled tetrahedron order. Source geometry and connectivity
were retained; no node separation or cavity repair was admitted.

The complete pinned case18 archive was re-imported locally after four missing
buffers were found in the retained scratch asset. Its SHA-256 matched the
[source pin](CARDIAC_WALL_ANATOMY_20260912.md), and all 13 imported buffer
hashes matched that asset's existing manifest. The full material-frame converter
then produced **1,470,083** source-ordered quaternions, with SHA-256
`9fb21ec06d6d73b209f47b1f905708b9ff26cb228114786fb3dbf9163e1ff0be`.
Its original source fields remain unchanged. The derived frame field and its
source-bound manifest are in the
[evidence directory](media/cardiac-ventricular-native-step-20260930/).

The cooked ventricular package has SHA-256
`6de9d9837d15d1f2bfd281b4dc65c6996d6eb4329490d2ed4fd9cb6c7bbf46d5`.
It is reproducible from the published input buffers and
[Numi Lab Matter `coupled` revision `5787938`](https://github.com/Numi2/numi-lab/commit/5787938325d6d232074bc508940c86c6b8b7a29d),
which adds the checked source-to-cooked map and the full-source cook/step
diagnostics. The 219 MB native package is retained in the local ignored Build
directory; its hash and exact cooking binary are pinned in `receipt.json`.

The physical M4 accepted **one 1 µs microstep** with the source-derived 100 ms
candidate tension. Two fresh active runs produced byte-identical complete
accepted node buffers. A zero-tension run from the same cooked package also
accepted. Compared with that control, **190,921 nodes** changed position;
the maximum active-minus-zero displacement was **4.88670199885e-7 m** and the
RMS difference was **1.32957106602e-8 m**. The mass fields matched bitwise and
the three fixed nodes did not move. The active state SHA-256 is
`7d9d89ff5fc9e5dc94ff9ab8be1ea8c338c575a0e81e857bacdbb35d86c8baa7`;
the zero state SHA-256 is
`ed6f820c13f6c3816ae8c542f5c80e040844e92677b0a1ea0feb86101a15f5f3`.
The active and zero accepted node buffers, input field, source frames, exact
binary hashes, run outputs, comparison, and receipt are published with the
[evidence](media/cardiac-ventricular-native-step-20260930/). The independent
gate rebuilt the package from those source inputs and repeated both active
runs plus the zero run against the exact retained state hashes.

This is a **bounded mechanics-ingress fixture**. It assigns synthetic
**1050 kg/m³ inertial density**, fixes only three nodes, treats the supplied
loaded CT mesh as the computational reference, and applies no chamber pressure,
valve or vessel port condition, blood-flow load, or measured motion target.
The other **372,549** nonventricular case18 tetrahedra were not cooked. The
candidate activation still differs from published CARP timing. Therefore this
one accepted ventricular step proves source-scale native ownership and replay,
not an anatomically supported contraction, a four-chamber simulation, energy
closure, or heartbeat qualification.

To verify the retained evidence, run
`python -m numilab_human.cardiac_ventricular_native_gate` with `--asset`,
`--activation`, `--tension`, `--evidence`, `--native-repo`,
`--cooked-package`, `--cook-binary`, `--step-binary`, and `--output` paths.
Add `--execute` to re-cook the 1,097,534-cell source subset and re-run the
three native transactions. The exact source archive URL, checksums, conversion
policy, and native revision are pinned in the linked manifests and receipt.
