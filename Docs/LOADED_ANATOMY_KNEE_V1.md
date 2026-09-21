# Loaded left-knee authoring v1

`HumanPack.loaded-anatomy-knee.v1` is the immutable Human-to-Lab authoring
boundary for the loaded left Open Knee `oks003` candidate. It binds one exact
source anatomy, projected reference, material policy, mass partition, contact
scope, tendon replacement program, and state-owner contract. It does not own or
publish an accepted runtime state.

## Compile the bundle

First compile the ownership prerequisite from a validated target-coverage
manifest and the checked-in left-knee declarations:

```sh
.numi/commands/human-ownership-compile \
  --coverage /path/to/current-target-coverage.json \
  --authoring config/human-ownership.loaded-anatomy-knee-left.v1.json \
  --output /path/to/HumanPack.ownership.v1.json
```

The Lab authoring export is the other prerequisite because it owns the body
poses and the source-to-reference mapping. Once both prerequisites exist,
compile the Human bundle with:

```sh
.numi/commands/human-loaded-anatomy-knee-compile \
  --ownership /path/to/HumanPack.ownership.v1.json \
  --open-knee-payload /path/to/open-knee-oks003-left-v4.nhknee \
  --tendon-payload /path/to/numi-human-tendon-attachments.nhtendon \
  --x-ref /path/to/loaded-knee-x-ref.f32le \
  --lab-export /path/to/loaded-knee-authoring-export.v1.json \
  --output-directory /path/to/loaded-knee-authoring-bundle
```

The equivalent routed command is `numi human loaded-anatomy-knee-compile`.
The checked-in left-knee profile is frozen inside the compiler; callers cannot
substitute a different profile.

The output directory contains exactly:

- `HumanPack.loaded-anatomy-knee.v1.json`, the complete authoring receipt; and
- `HumanPack.loaded-anatomy-knee.binding.v1.json`, an exact embedded-manifest
  envelope for one-step Lab admission.

Both files are canonical JSON with internal identities. They are staged,
flushed, and published together by one directory rename. Reusing the directory
is idempotent only when both files are byte-identical; different content or an
unexpected file fails closed.

## Frozen source and execution order

The anatomy input is the left `NHKNEE1` ABI3 payload with SHA-256
`2e38201528de25911ea496164602ea7e823cd9c2e3c94efa550ee52689324ae5`
and 34,357,400 bytes. The selected runtime order is permanently
`ACL, LCL, MCL, PCL, PTL, QAT`, not the payload's incidental region-table
order. That order covers 62,402 nodes and 264,442 tetrahedra.

The current tendon input is `NHTENDON3` with SHA-256
`72ca0ec4ef53f647784a89f761e78495d4997c7d72cfe6a1d9eedb24dd97feaa`
and 238,288 bytes. It binds all 832 endpoints and the exact four quadriceps
replacement tuples for `recfem_l`, `vasint_l`, `vaslat_l`, and `vasmed_l`.
Historical tendon payloads are not admitted by this contract.

The authoring export and Human receipt jointly bind:

- the source-global node index stream;
- executable remapped tetrahedra in source-element order;
- source anchor ownership and executed moving-enthesis anchors;
- per-region runtime node and tetrahedron spans;
- raw float32 lumped node masses;
- the exact rigid and equality payload identities;
- source and projected body-pose identities; and
- donor source, subtracted, remaining, and closure moments.

The five passive regions are `ACL`, `LCL`, `MCL`, `PCL`, and `PTL`. The seven
articular pairs retain their full ABI3 source order:
`TBC-L_To_FMC`, `TBC-L_To_MNS-L`, `PTC_To_FMC`, `MNS-L_To_FMC`,
`MNS-M_To_TBC-M`, `MNS-M_To_FMC`, and `TBC-M_To_FMC`. Femur body 145 donates
mass to `ACL`, `LCL`, `MCL`, `PCL`, and `QAT`; tibia body 150 donates mass to
`PTL`; patella body 156 is explicitly excluded as a mass donor.
The frozen profile keeps the readable nominal donor masses `8.4 kg` and
`3.8 kg`; the source-derived export and authored receipt preserve their exact
EngineModel float32 values `8.399999618530273 kg` and
`3.799999952316284 kg`. Human requires that exact nominal-to-float32 conversion,
not a tolerance.

## Coordinate and state ownership

The coordinate chain has four distinct states:

1. `A`, or `x_source`, is ABI3 `restWorld` in MyoSim world metres, registered
   against the unprojected default body pose.
2. `B_ref`, or `x_ref`, maps `A` from the unprojected default bodies to the
   equality-projected default bodies. Human authoring owns this immutable
   projected-rest candidate.
3. `C_init` maps `B_ref` from the projected reference bodies into the support or
   qualification pose. Lab runtime owns this initialization state.
4. `x_current` is the accepted runtime position state. Only the separate Lab
   accepted-step transaction may publish its hash.

The compiler independently recomputes signed source-to-reference tetrahedron
Jacobians. A collapsed or orientation-reversing element is rejected. None of
`A`, `B_ref`, `C_init`, or `x_current` is described as unloaded or stress-free.
The reference class is `projected-rest-candidate`; prestress reset remains
unresolved and volumetric prestress is not applied.

The frozen A-to-B_ref construction uses
`adaptive-dyadic-first-success-slerp-geodesic-inverse-distance-moving-enthesis.1`.
Its implementation identity is encoded as
`sha256(domain-utf8||header-file-sha256||core-file-sha256||adapter-file-sha256)`:
the UTF-8 domain `numi.lab.loaded-knee.a-to-b-ref-code-identity.v1`, followed by
the raw 32-byte SHA-256 digests of the continuum-map header, core, and probe
adapter in that order.
It transports the exact bone-owned entheses along linear translations and
shortest-arc quaternion interpolation, retrying the complete source mesh with
1, 2, 4, ... increments until every incremental tetrahedron passes the signed
Jacobian gate. It always reaches the exact projected body poses; partial target
motion, determinant clamping, and vertex nudging are not admitted. The frozen
PTL source rejects the direct one-increment interpolation at local tetrahedron
419 and succeeds with two increments. Failure to find an admitted dyadic path
fails closed.

The Lab export records one exact diagnostics row for each region in
`ACL, LCL, MCL, PCL, PTL, QAT` order. Human rechecks each row against the
persisted float32 `B_ref`: per-region and aggregate signed Jacobian extrema must
match Human's independent recomputation, the direct-map status and selected
increment count are frozen, and source, final-double, and persisted-float32
anchor residuals must each be no more than 200 nanometres. This continuation is
a deterministic geometric authoring map, not a material equilibrium solve, an
unloaded reference, or evidence of physiological prestrain.

The executable nodal-mass digest uses the pinned algorithm
`matter-referenced-f32-volume-f32-density-fp64-source-order-accumulate-final-f32.1`.
For each tetrahedron in source element order, Human rounds the positive
reference volume to IEEE float32 exactly as Matter stores `inverseRestRow0.w`,
multiplies its double promotion by the double promotion of float32 density
`1000`, contributes one quarter to each node, accumulates each node in float64,
then rounds the ordered nodal results to little-endian float32 for hashing.
Physical mass and first/raw-second moments remain independent double-volume
integrals; the executable hash does not lower their precision.

## Evidence boundary

Top-level `status: candidate` describes this scoped authoring artifact. It does
not erase `source_ownership_status`, which remains `blocked` or `partial` from
the wider ownership graph.

Lab admission must also load the exact canonical `HumanPack.ownership.v1`
artifact named by `inputs.ownership`, verify its full-file and internal manifest
SHA-256 identities, derive the source status and selected coverage leaves from
its records, and preserve both unchanged. The binding alone cannot authenticate
those external ownership facts.

A Lab run may consume the bundle for explicitly
non-production candidate mechanics, but it must preserve that source status and
must keep all of the following false:

- unloaded-reference and prestress-equilibrium qualification;
- subject-specific material calibration;
- mesh-convergence and specimen-load qualification;
- clinical validity;
- production physical or active-force ownership;
- a global seven-owner accepted-state root; and
- integrated-Human qualification.

Materials and density are source population priors. Successful compilation is
authoring evidence only. Physical M4 execution must separately verify the exact
binding and input bytes, executable topology and node masses, donor
mass/moment closure, actual contact response, all declared mutable state,
eight accepted 50-microsecond steps, exact restore/replay, and unchanged state
after a rejected attempt. Even that result remains a scoped physical candidate,
not physiological, clinical, performance, or production qualification.
