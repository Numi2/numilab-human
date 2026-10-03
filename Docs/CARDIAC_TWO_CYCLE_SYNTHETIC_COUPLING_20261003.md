# Native two-cycle synthetic heartbeat coupling — 3 October 2026

The current Matter heartbeat checker completed a single preregistered native
Metal run on Apple M4. It used a synthetic 64-node, 156-tetrahedron hollow
shell, two matched environments (periodic active tension and zero-tension
control), three hydraulic compartments, two directional valves, and a
synthetic return resistor. Both environments accepted 400 clock steps at
4 ms, covering two 0.8 s cycles. The active-tension pulse was prescribed by
the fixture; no electrical activation field drove it.

Both active cycles ejected approximately `8.5153e-8 m³`. Recomputed from the
retained per-step trace, cycle-to-cycle ejection drift was `1.43e-6` relative;
filling differed from ejection by `0.4952%` and `0.00261%` across the two
cycles; systemic return differed from ejection by `0.04447%` and `0.04431%`.
Maximum total blood-volume error was `5.59e-6`. Active and control hydraulic
cavity volumes tracked their respective shell volumes within `1.59e-7` and
`2.08e-7` relative to the fixture's `1e-6 m³` scale. Active wall displacement
reached `4.98e-4 m`; the zero-tension control reached `2.65e-6 m`.

The full step loop took `113.1 s` for the 400 paired clock submissions. This
timing is retained for provenance only and is not a performance qualification.
The independent CSV audit verified 400 finite rows, monotonic time through
`1.600000076 s`, a mean timestep of `0.004000000190 s`, conservation, both
beat closures, and repeatability. The run passed every threshold sealed in the
[preregistered plan](media/native-synthetic-heartbeat-coupling-20261003/preregistered-plan.json).
The [raw trace](media/native-synthetic-heartbeat-coupling-20261003/heartbeat-trace.csv),
[stdout](media/native-synthetic-heartbeat-coupling-20261003/stdout.txt),
[independent audit](media/native-synthetic-heartbeat-coupling-20261003/csv-independent-audit.json),
[run manifest](media/native-synthetic-heartbeat-coupling-20261003/run-manifest.json),
and [file checksums](media/native-synthetic-heartbeat-coupling-20261003/SHA256SUMS)
preserve the result.

The checker source was a pre-existing uncommitted change in Matter commit
`5bf85aed50cf9af1d55445ac38b2a2bc71b1dcfe`; it was rebuilt from the exact
working-tree file whose SHA-256 is recorded in the manifest. This run confirms
the added synthetic return-flow and repeated-beat assertions on that source
snapshot. The native source working file remains uncommitted and was not
modified by this verification.

This is **synthetic native heartbeat mechanics evidence** only. It does not
qualify anatomical myocardial geometry, electrical conduction, ECG, source
activation timing, physiological calibration, biological or clinical validity,
or integrated whole-Human capability. The synthetic wall, prescribed pulse,
hydraulic reservoirs, and valve ports are not anatomically assigned. The
existing [source-driven 0D cardiac model](CARDIAC_HEARTBEAT_20261002.md),
[source-derived tension sequence](CARDIAC_NATIVE_ACTIVE_TENSION_SEQUENCE_20261002.md),
and this two-cycle shell fixture remain separate evidence lineages. The next
cardiac gates remain reaction-eikonal timing on the accepted native clock,
source-defined valve semantics and anatomical port binding, then coupling to
source myocardium with independent electrical and mechanical controls.
