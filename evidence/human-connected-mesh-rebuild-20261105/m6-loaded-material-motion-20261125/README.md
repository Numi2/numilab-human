# Loaded skin and fat on SSH Numi

The dense native owner imports 314,014 nodes and 1,457,316 skin and fat tetrahedra. The sparse update changes 453 loaded nodes. The stress-free positions, tissue identities, 14 deep attachments, 4,733 cortical binding facets and physical mass tables match the parent.

The whole-volume native-coordinate audit found 626 cells outside the project's principal-stretch interval of 2/3 to 1.5 in the previous registration and 314 in this one. Both registrations have positive cells. The repair reduced stored elastic energy from 4.149 J to 2.744 J and conserved 3.452571 kg of skin and 10.487108 kg of fat. The tissue still contains 325 sampled cortical points. For the affected cortical banks the measured overlap decreased by 33.168 mm³. No new cortical triangle pairs were detected.

Thirty-three remaining cells violate a stretch bound even on their fixed vertices. Those locks come from the repair's collision safeguards. The 14 anatomical attachments impose none of those 33 restrictions. The largest connected group of remaining failures contains 259 cells.

On SSH Numi, check the pinned dense owner with `scripts/run_resting_human_on_numi.sh --dense --check`. Use `--dense --steps 2` for the matched native motion trial. The checkpoint uses a 7.8125 µs physical step and variable pleural pressure. The existing `--current` run uses regional bed loading. The full-volume stretch and occupancy check is available with `--current --material-report /tmp/numi-loaded-material.json --material-proposal PATH/loaded-registration.npz` and exits unsuccessfully for the known defects.

The recovery manifest identifies the independently restored native binary, material source, corrected coordinates, build inputs and numerical checks. This improvement remains an experimental registration.
