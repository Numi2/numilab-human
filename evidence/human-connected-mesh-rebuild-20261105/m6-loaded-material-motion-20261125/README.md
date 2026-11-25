# Loaded skin and fat on SSH Numi

On SSH Numi, `bash scripts/run_resting_human_on_numi.sh` replays the two-step integrated release with the full registered skin/fat volume, anatomical joint stops, the physical bed and variable pleural pressure. Use `--check` to verify its 564 pinned inputs. The explicit `16` and `60` durations use this same physical scene. The regional isolation run remains available under `--regional` or its existing `--current` alias. The current integrated release still starts out of equilibrium.


The dense native owner imports 314,014 nodes and 1,457,316 skin and fat tetrahedra. The larger experimental registration changes 453 loaded nodes. The stress-free positions, tissue identities, 14 deep attachments, 4,733 cortical binding facets and physical mass tables match the parent.

The whole-volume native-coordinate audit found 626 cells outside the project's principal-stretch interval of 2/3 to 1.5 in the previous registration and 314 in the larger repair. Both registrations have positive cells. The repair reduced stored elastic energy from 4.149 J to 2.744 J and conserved 3.452571 kg of skin and 10.487108 kg of fat. The tissue still contains 325 sampled cortical points. For the affected cortical banks the measured overlap decreased by 33.168 mm³. No new cortical triangle pairs were detected.

Thirty-three remaining cells violate a stretch bound even on their fixed vertices. Those locks come from the repair's collision safeguards. The 14 anatomical attachments impose none of those 33 restrictions. The largest connected group of remaining failures contains 259 cells.

## Dense native motion on Apple M6

Run `scripts/run_resting_human_on_numi.sh --dense --check` on SSH Numi to verify the pinned native inputs. Run `--dense --steps 2` to replay the bounded registration with both conservative dermal cavities. This is a 7.8125 µs physical timestep with variable pleural pressure. The existing `--current` scene has regional bed loading.

| Dense registration | Initial cells outside stretch bounds | Accepted physical time | Endpoint cells outside stretch bounds | Endpoint cells below shape bound | Interior arm cortical samples | Wall time including startup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous control | 626 | 15.625 µs | 567 | 37 | 908 | 215.71 s |
| Bounded repair, original connectivity | 569 | 15.625 µs | 568 | 37 | 908 | 219.31 s |
| Bounded repair and both dermal cavities, replayed by `--dense` | 569 | 15.625 µs | 563 | 37 | 908 | 215.78 s |
| Larger repair, original connectivity | 314 | 0 | — | — | — | 197.70 s |

The three completed runs had zero inverted cells and four cells outside the relative-volume bounds at the endpoint. The bounded repair improved the starting configuration and showed no measurable improvement in accepted tissue fidelity. The original dense control remains the baseline for comparing physical accuracy. Cortical samples count the tested bone vertices and face centres strictly inside tissue across eight upper-limb bones. Multiple samples can belong to the same crossing. The larger repair failed the 5 MPa skin material guard at local tetrahedron 132817. The failed launch is retained as `larger-repair-launch.json.gz`.

Use `--dense --material-report /tmp/numi-material.json --accepted-run ABSOLUTE_RUN_DIRECTORY` to audit the matched terminal FEM nodes and rigid poses. It verifies the accepted render pack, reads the compensated native coordinates at their exported precision, reproduces the two material shape counts from native output, and exits with failure when physical defects remain. To check a prepared registration before a run, use `--current --material-report /tmp/numi-opening.json --material-proposal PATH/loaded-registration.npz`.

The JSON comparison records input and output checksums and separate physical and wall times. The independently restored R4 and R5 archives retain the launches, source, dense registrations, native observations and failed trials. Neither two-step replay establishes supported settling, sustained throughput or continuous physiology.

## Conservative dermal remeshing

The first stress rejection occurred in a 6.83 nm-high reference tetrahedron. Remeshing its connected cavity replaced 30 tetrahedra with 27 and raised the minimum reference altitude to 5.70 µm. The native check measured a 2.389-fold increase in minimum shape quality. The mesh retained the oriented outer, dermal and cortical faces. All 4,733 cortical facet registrations passed. Transport used the existing quadrature radius and original material carriers. The registered donor positions now pass from the source cook to remeshing. Total loaded mass changed by 3.20 × 10⁻²⁰ kg. The largest donor first-moment change was 5.42 × 10⁻²⁰ kg·m.

The matched native trial rejected step one with 5,000,479 Pa in another dermal tetrahedron. Its source index was 1439803. The prior witness was 132817. They share donor 43, region 20 and source face 10134. Both have all four vertices on the skin–fat interface. Their centres are 6.63 mm apart. All 1,457,316 original cells were surveyed. There were 77,040 with minimum reference altitude under 1 µm, including 42 of the 568 endpoint stretch failures. 526 of the 568 failing cells were outside that thin-cell subset. These counts establish co-location only.

The two completed control runs used dynamic release. Initial maximum acceleration residual was 68.31 m/s². The last accepted root carried approximately 11.01 N through the physical skin–bed reaction on a 97.13 kg rigid-plus-FEM assembly. No floating-root equilibrium check ran during initialization. These records establish contact and provide a quantitative deficit for supported settling.

The native source patch applies to the exact input hashes in `native-dermal-source.json`. The R6 recovery manifest binds the full tested source and binary, conservative connectivity, build inputs, exact failure witnesses and preserved rejected trials. The geometric and mass checks passed in the native owner. The 30-to-27 topology remains experimental.

## Both native stress witnesses

The authenticated donor and source region are shared, with two disconnected pieces of dermis. A boundary-preserving rebuild of the second piece kept 30 of its 58 source tetrahedra exactly. The other 28 were replaced with 31. On that changed portion the minimum reference shape improved from 1.657 × 10⁻⁵ to 5.412 × 10⁻⁵. Minimum altitude across the entire piece increased from 2.99 nm to 448 nm. Density stayed at the original 900 kg/m³. Both repairs together retained 1,457,316 cells, all 4,733 cortical registrations and the original source donors. Native loaded mass drift was 2.99 × 10⁻²¹ kg.

With the larger forearm registration, the full native trial reached a different failure on the first step. All 22 object certificates recorded the same whole-system nonlinear residual of 0.031993. The configured Newton target was 0.000010. There were 16 active Newton iterations and 298 active Krylov columns. No inner solve exhausted its budget. Neither dermal stress witness recurred as the first rejection. This run accepted zero steps.

The executable `registered_cavity_geometry` in the independently restored R7 archive calls the same reference-geometry predicates as the full native loader. The source and `check_cavity_geometry.py` are under `tools/native_ligament_contact/registered_material_volume`. Give the Python tool the emitted TetGen nodes, elements, NPZ proposal and compiled executable using `--nodes`, `--elements`, `--proposal` and `--executable`. It reproduces the native rejection for an unnecessarily large 58-cell input and passes both compact repairs. Checking both pre-extracted cavities took 4.9 ms median across five repetitions. This timing excludes file preparation and applies only to the local reference geometry checks.

## Matched motion and supported-rest decision

The current dense launcher uses the previous moving registration's byte-identical opening tissue coordinates and both conservative dermal changes. The native material and rigid-body capture reached 15.625 µs. At that step there were 563 stretch failures, four relative-volume failures, 37 low-quality cells, zero inverted cells and 908 sampled cortical points inside skin or fat. The change cleared five stretch failures in aggregate compared with the identical starting registration on its previous connectivity. The one measured post-startup interval ran at 2.05 × 10⁻⁷ real time. The complete two-step trial including startup took 215.78 s. This is an experimental motion checkpoint. Neither a breath nor a filling–ejection cycle finished in this duration.

At the starting pose the gravity line projected 28.57 mm inside the hull of eligible skin–bed contacts. The required ratio of tangential to normal support is 0.00251 on the slightly tilted bed. The logged rigid-material static friction coefficient is 0.05. The footprint permits the required centre of pressure. The 97.13 kg assembly weighs 952.86 N in its native gravity field. Its final accepted FEM bed reaction averaged 10.99 N over the last root. Initialization gave a 68.31 m/s² maximum acceleration residual. The primary physical deficit is the load the tissues and skeleton carry at the starting equilibrium. To reproduce the footprint and gravity calculation, run `python3 check_bed_support.py NATIVE_RUN_DIRECTORY support.json` with the restored native inputs. The calculation uses initial geometry and the gravity vector recorded in the rigid pack.

The same two cavities on the larger-displacement registration previously failed the coupled residual at root zero. Transplanting its interior-node corrections into the bounded registration also failed the loaded first-moment check. Restoring the exact bounded coordinates passed this check with the existing tolerance. Source, failed attempts, successful native binary, full synchronized capture and independently restored file digests are retained in the R8 archive.
