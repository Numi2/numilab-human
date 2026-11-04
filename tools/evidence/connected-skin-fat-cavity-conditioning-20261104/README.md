# Connected skin/fat conditioning: exact geometry and conservative transfer

This is a local connected volume repair of the existing registered mixed-source
reference adult. No surface vertex, layer boundary, material identity, anatomical
attachment or source position is moved. This is not a measured single individual.

The proposal retains 41,391 accepted legacy 3-to-2 cavities and adds 468 bounded
connected cavities. It changes 1,453,640 skin/fat cells to 1,453,060. The native
importer certifies exact Float32 boundary/volume and positive cells; 11 cavities
use the existing conservative transport owner to route their existing parent
sources within the unchanged 0.187444 m radius. Infeasible candidates remain
baseline cells. Complete source ancestry, donor/region/material IDs and nodal
mass accounting are retained.

Source volume SHA-256:
`5d31c141b66628801a711c2e745f744f745611882a269e173eddfcd0e99d3372`.
Final proposal SHA-256:
`4be98e7d5710360407a561d10efaa02b86702efdaf1b4312c4b035d6d9a19cb8`.
The report records the immutable accepted base and complete old/new cell ancestry.
`replay_connected_cavities.py` reproduces the proposal from that base and report
with its hash checked; it does not run another geometry solver or introduce an
artifact format. Executed construction/filter scripts and unsuccessful proposal
counts are retained separately.

Native admission and source recovery are published with the owner implementation:
[Numi Lab 9ced64691](https://github.com/Numi2/numi-lab/commit/9ced64691).
The existing checker passes 29 negative cases, legacy controls, exact connected
geometry, conservative heterogeneous mass, restricted routing and atomic failure.
All 41,859 final cavities pass the native geometry checker. The exact construction
replay and all 39 execution source pins pass.

Native135 accepts 24 steps at 15.625 microseconds, then rejects step 25 with a
nonlinear solver failure. The previous geometry stopped after 20 accepted steps.
The accepted interval is only 0.375 ms of the release transient. This does not
qualify motion, resting equilibrium, a breathing cycle or end-to-end performance.
See native135-summary.json for timings and remaining failure. Material laws,
contact limits, timestep and residual acceptance tolerances are unchanged.

The retained earlier failed trials remain evidence in the linked Lab increment.
This candidate is not promoted to the accepted motion-qualified scene.
