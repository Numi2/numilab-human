# Bilateral extensor digitorum under supported native motion

The tested NHTISS4 registration rotates existing donor frames about their named MyoSim sites. A donor with two distinct sites can rotate about their line. Three noncollinear sites fix the frame. Sixteen of the nineteen donor frames change per side. The 72 named site entries remain within 0.100 µm after serialization. Source positions, topology, skinning weights, masses and actuator routes remain identical. These banks represent the passive muscle envelopes. The existing muscle and tendon owners produce the forces.

The authoring fit used the compact entrance and 16-second terminal poses. Two independently initialized 8-second native runs tested a pose excluded from the fit. Each comparison checked topology, all 877 other geometry banks, articulated poses, and complete physiological and support-impulse traces. Counts below are intersecting triangle pairs summed over both extensors.

| Native pose | Self pairs before → after | Against skin | Against bone | Against neighboring muscles |
|---|---:|---:|---:|---:|
| Compact entrance | 717 → 575 | 840 → 609 | 1,221 → 1,098 | 1,030 → 630 |
| Compact 64 ms | 722 → 575 | 831 → 616 | 1,220 → 1,097 | 960 → 487 |
| Compact 8 s | 835 → 708 | 1,275 → 1,113 | 1,631 → 1,307 | 2,519 → 1,830 |
| Compact 16 s | 881 → 676 | 1,235 → 1,121 | 1,660 → 1,364 | 2,782 → 1,887 |
| Dense 31.25 µs | 717 → 575 | 2,386 → 1,565 | 1,221 → 1,098 | 1,028 → 630 |

No previously clear anatomical bank acquired crossings at these snapshots. Several already affected banks worsened. At eight seconds the right extensor gained 102 pairs with muscle 113. The compact whole-body endpoint had 13,950 nonocular skin/anatomy pairs and 40 skin self-intersections. The 8,000 accepted steps used variable pleural pressure, with three complete breaths and 19 filling/ejection cycles. Body runtime was 156.938 s (0.10195×) and cold runtime was 165.310 s (0.09679×). The run used 32-step presentation cadence and discrete geometry captures.

The dense native run retained identical skeletal and respiratory states, contact-impulse traces and physical skin/fat coordinates. All 705,090 skin cells and 743,709 fat cells passed the existing shape and orientation gates at its two-step endpoint. The three material boundaries had zero mutual and self crossings under the installed-layer test. The full skin/anatomy test still failed with 10,970 pairs. This dense run simulated 31.25 µs in 205.83 seconds cold. It does not establish settled equilibrium or respiratory motion for that model.

**Regional rejection.** Dense right outer/inner/deep intersections fell from 1,794/1,633/1,187 to 915/1,577/1,156. The left changed from 592/1,271/1,187 to 650/1,310/981. The left outer and inner regressions disqualify this fit as the common bilateral registration. This is an experimental checkpoint with recoverable inputs and emitted output.

## Reproduction on SSH Numi

Use the existing native Human asset owners and the parent recovery receipts in the sibling supported-bed-footprint evidence. Recover the retained files using the two JSON receipts here. They record the member hashes verified by independent download. On Numi the Python module search path includes the Human tools for registered digit material, posterior registration, and the original native envelope audit. The native geometry reader, route registry and exact pair predicate used by the fit are in the archive.

```sh
python3 solve_extensor_sites.py "$entrance_case" "$terminal_control" "$reader" "$routes" "$predicate" "$fitted"
python3 install_extensor_bindings.py "$drive_control" "$fitted" "$routes" "$registered_parent"
NUMI_RETAIN_SUPPORT_GEOMETRY=1 NUMI_CAPTURE_BED_COLUMNS=1 python3 run_bed_work.py "$registered_parent" "$footprint_owner" "$run" 8000 --disabled
python3 verify_native_extensor_pair.py "$terminal_control" "$run" "$reader" "$predicate"
python3 verify_native_extensor_pair.py "$dense_control" "$dense_trial" "$reader" "$predicate" --dense
python3 verify_extensor_material_layers.py "$dense_control" "$dense_trial" "$reader" "$predicate"
```

The final native-run switch selects the configuration without the optional thoracic diagnostic. Its resting control, mechanical respiration and circulation remain active. The binary, device and input pins, launches, failed alternatives, comparison counts per anatomical identity, trace hashes, and compressed emitted geometry are retained. Anatomy represents a registered reference assembly assembled from multiple sources.
