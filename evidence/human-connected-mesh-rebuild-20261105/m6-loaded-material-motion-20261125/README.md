# Loaded skin and fat on SSH Numi

The dense native owner imports 314,014 nodes and 1,457,316 skin and fat tetrahedra. The larger experimental registration changes 453 loaded nodes. The stress-free positions, tissue identities, 14 deep attachments, 4,733 cortical binding facets and physical mass tables match the parent.

The whole-volume native-coordinate audit found 626 cells outside the project's principal-stretch interval of 2/3 to 1.5 in the previous registration and 314 in the larger repair. Both registrations have positive cells. The repair reduced stored elastic energy from 4.149 J to 2.744 J and conserved 3.452571 kg of skin and 10.487108 kg of fat. The tissue still contains 325 sampled cortical points. For the affected cortical banks the measured overlap decreased by 33.168 mm³. No new cortical triangle pairs were detected.

Thirty-three remaining cells violate a stretch bound even on their fixed vertices. Those locks come from the repair's collision safeguards. The 14 anatomical attachments impose none of those 33 restrictions. The largest connected group of remaining failures contains 259 cells.

## Dense native motion on Apple M6

Run `scripts/run_resting_human_on_numi.sh --dense --check` on SSH Numi to verify the 476 pinned native inputs. Run `--dense --steps 2` to replay the registration whose complete two-step capture is archived. This is a 7.8125 µs physical timestep with variable pleural pressure. The existing `--current` scene has regional bed loading.

| Dense registration | Initial cells outside stretch bounds | Accepted physical time | Endpoint cells outside stretch bounds | Endpoint cells below shape bound | Interior arm cortical samples | Wall time including startup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous control | 626 | 15.625 µs | 567 | 37 | 908 | 215.71 s |
| Bounded repair, replayed by `--dense` | 569 | 15.625 µs | 568 | 37 | 908 | 219.31 s |
| Larger repair | 314 | 0 | — | — | — | 197.70 s |

Both completed runs had zero inverted cells and four cells outside the relative-volume bounds at the endpoint. The bounded repair improved the starting configuration and showed no measurable improvement in accepted tissue fidelity. The original dense control remains the baseline for comparing physical accuracy. Cortical samples count the tested bone vertices and face centres strictly inside tissue across eight upper-limb bones. Multiple samples can belong to the same crossing. The larger repair failed the 5 MPa skin material guard at local tetrahedron 132817. The failed launch is retained as `larger-repair-launch.json.gz`.

Use `--current --material-report /tmp/numi-material.json --accepted-run ABSOLUTE_RUN_DIRECTORY` to audit the matched terminal FEM nodes and rigid poses. It verifies the accepted render pack, reads the compensated native coordinates at their exported precision, reproduces the two material shape counts from native output, and exits with failure when physical defects remain. To check a prepared registration before a run, use `--material-proposal PATH/loaded-registration.npz` in place of `--accepted-run`.

The JSON comparison records input and output checksums and separate physical and wall times. The independently restored R4 and R5 archives retain the launches, source, dense registrations, native observations and failed trials. Neither two-step replay establishes supported settling, sustained throughput or continuous physiology.
