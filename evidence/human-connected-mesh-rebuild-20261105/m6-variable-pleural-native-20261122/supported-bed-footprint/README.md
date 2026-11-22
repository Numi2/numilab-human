# Bed load and anatomical clearance on the supported native Human

The compact supported model completed 8,000 accepted 2 ms steps on SSH Numi’s Apple M6. The native body run took 155.774 s for 16 simulated seconds (0.10271×). Cold launch through native return took 164.079 s (0.09751×). Accepted-state presentation and telemetry ran at a 32-step cadence, averaging 1.60 callbacks per wall second. The 15 Hz presentation target remains open.

This revision changes the area used to assemble physical bed stiffness. It integrates skin shape functions over the lowest projected surface visible from the flat bed, within each section’s original 20 mm reference band. A grid limits triangle comparisons at scene preparation. Contact during motion still uses the existing Metal solver, selected skin positions and their articulated Jacobians. The area is frozen at initialization. The two effective constitutive parameters remain 200 kPa for skin and 55 kPa for fat.

The signed change in effective footprint was 0.199474 to 0.195155 m² over the existing 32 sections. The new calculation also cuts triangles at the depth limit. Across equal time samples in the final two seconds the mean bed reaction changed from 952.73 to 952.92 N. The whole-body weight is 952.86 N. This measures vertical support. The dense reference still has a tissue-to-skeleton equilibrium deficit.

Variable pleural pressure was active, ranging from −770.01 to −470.34 Pa at retained samples. The terminal trace recorded three complete breaths, 19 complete filling/ejection cycles, a 528.82 mL tidal volume and a 69.29 mL left ventricular stroke volume. Terminal arterial O₂ was 102.71 mmHg, CO₂ was 38.95 mmHg and the reported blood volume error was 0.0042 mL. These simulated outputs do not establish clinical validity.

| Native pose | Skin–anatomy triangle pairs | Bone subset | Skin self-intersections |
| --- | ---: | ---: | ---: |
| Before the first physical step | 8,859 | 1,760 | 17 |
| 0.064 s | 8,760 | 1,774 | 17 |
| 16 s | 14,064 | 2,791 | 40 |

All three audits used frozen native body transforms, 108,987 skin faces and 878 anatomy banks. The observed startup build captured all 157 initial body poses. Its 7,273 section vertices exactly coincide with the emitted initial skin, and their areas reproduce the native sections. The startup observer and independently rebuilt binary with the same physical model reproduced the normal step-32 pack and the body and respiratory state hashes. The initial audit found 6,541 skin–muscle triangle pairs, rising to 10,953 at 16 s. Across all physical banks, 1,156 exact triangle-pair identities persisted from the initial to the final frame. Changes of triangle identity can occur inside the same region of overlap. These are discrete geometry checks.

Retain this footprint correction for the compact load path. Reject whole-body anatomical qualification and the promotion of the compact model as a substitute for the dense skin/fat reference. The next registration and mechanical coupling changes must be evaluated on the same loaded pose and through motion. A bed-stiffness change cannot resolve 8,859 crossings present before stepping. The separate dense load witness, including measured skeleton and bed forces, is in the neighboring paired-fat-registration/supported-material-strain evidence.

The two recovery receipts identify the downloadable source, tested binaries, native geometry, mode and exact input hashes. Both archives passed independent download and per-file SHA-256 restoration. The native owner build starts from the existing registered drive-heart build. On Numi the commands used the pinned working Apple developer tools, the existing app build/native object dependencies and the recovered Human inputs.

```sh
python3 build_bed_footprint.py "$drive_heart" "$app" "$owner"
NUMI_RETAIN_SUPPORT_GEOMETRY=1 python3 run_bed_work.py "$drive_control" "$owner" "$trial" 8000 --disabled
python3 check_drive_anatomy.py "$trial" "$retained_surface_reader" "$registered_envelope_reader" --require-clear
```

The final switch disables the optional thoracic diagnostic capture. Native breathing, Brain control, circulation, contact and geometry certification ran together. For the independent startup audit, build with `--with-registration-capture` and run 32 steps with `NUMI_CAPTURE_BED_COLUMNS=1` and geometry retention. `check_initial_bed_anatomy.py` checks this pack against the loaded physical footprint.


## Why the forearms need a registration repair

An independent exact-plane check classified all 8,859 initial skin/anatomy pairs and all 14,064 at 16 seconds as transverse. In each pair both triangles have vertices strictly on both sides of the other triangle’s plane. Tangential contact and coplanar seams cannot explain these counts.

The right and left extensor digitorum have stable tissue IDs 137 and 138. The installed NHTISS4 source surfaces have zero self-intersections after exact seam welding. Applying their actual binding weights and the 19 native body transforms per muscle reproduced the native forearm envelopes to within 0.19 µm in the two captured poses. The initial registered muscles contain 359 and 358 self-intersecting face pairs. These rise to 460 and 421 by 16 seconds. The folded surfaces prevent interpreting them as clean solids for signed penetration or bed-compression estimates.

Repair the source-to-body mappings across the wrist and four digit branches, and test the linked muscles, tendons, bones and skin in the same actual loaded motion. Preserve the named muscle force routes and endpoints. A previously tried branch-weight fit retained substantial intersections. Its measured results are in the sibling registered-digital-material/diagnosis evidence. The dense tissue-to-skeleton support imbalance is a separate mechanical failure. Keep the reduced and dense configuration IDs alongside their respective results.

The compact endpoint audit now supports the same explicit clearance request as the dense audit. The documented command exits unsuccessfully for this 16-second result and retains the witnesses. Its failure message and the bilateral binding replay are in this directory. This endpoint check covers external skin versus nonocular physical banks and skin self-intersections.
