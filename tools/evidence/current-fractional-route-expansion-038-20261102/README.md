# Current-Skin whole-cut witness expansion (038)

This scoped source-and-evidence package records the current out-007 witness-input preparation and its actual full allocator diagnostic. It contains compact producer sources, the 300,860-row route proposal NPZ, the pinned 032d carrier packet, source-only census and append receipts, launch-preflight receipts, and compact actual-run/cut metadata. The 933,678,749-byte Skin CSV and 376,396,775-byte compressed full-cut witness remain at their pinned host paths and are not copied here.

The route census reports 72,675 of 355,193 T-Skin children with at least one new supported group route and 282,518 without one. The appended Skin CSV has 300,860 distinct group-child witnesses; its v15 prefix is byte-identical, and the Fat CSV and 032d packets are unchanged. The append replay checked source packet parent ownership through parent_offsets, child rows/centroids, one-hop P1 region support, and the pinned 0.187444 m envelope.

The actual out-007 run completed in 640.404 seconds without timing out, but returned exit 6 with INFEASIBLE_FULL_GRAPH_CUT. It sent 12.979344843761043 kg against a 12.998725017533202 kg minimum-density demand, a shortfall of 0.019380173772159 kg (about 19.38 g). The final cut lists 60,604 residual child rows totaling 0.019380173772254032 kg. This proves the current supported graph did not meet all minimum demands; it does not establish infeasibility under other valid support. The full cut witness is pinned by SHA and size in evidence/actual-out-007-run-result.json; only its compact header and manifests are copied.

No source-only mass admission was produced, and no physical admission or native steps occurred. The source geometry and native solver gates were not changed by this witness expansion. The launch preflight alone is PASS_LAUNCH_READY_NOT_RUN; it is not the allocator result.

Earlier report-serialization, source-crosswalk, and receipt-recovery attempts remain listed with source pins in evidence/attempt-history.json. The generator regression exercises self-template generation from out-007, whose config is named run-config-038.json; both that suffix and legacy run-config.json are accepted. The CSR control maps all 300,860 flattened carrier rows to owning parent tets, including zero-carrier parent rows.

Run focused controls from this directory with the project Python environment:

~~~sh
python -m pytest -q -p no:cacheprovider tests
~~~

The tests validate source pins, compact route fixtures, CSR replay, and the retained failed actual-run status. They do not rerun the allocator or qualify anatomy or physical mass transfer.
