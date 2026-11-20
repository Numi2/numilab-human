# Posterior pelvic clearance and native skin–fat mass transfer

This increment uses the [previous posterior fit](../posterior-load-interface/README.md) as control.
A connected deep-fat patch beside left iliac bank 128/19 clears 28 exact
crossing pairs across three faces. Four deep-node corrections are added to the
control's 187. The outer skin, bonded skin–fat interface, cell connectivity
and donor frame identities match the control. The independent startup audit
finds zero new bone, muscle-source, other FEM or shell pairs over changed faces.

The existing per-cell mass owner consumes 31,291 routes and conserves
13.939679838742 kg across 628 material/region/cell-donor pools. The maximum
source mass error is 1.74e-18 kg, and the longest route is 74.875 mm.
Minimizing the maximum skeletal donor debit change reduced it from 10.394 g
to 4.259 g. Per-link mass repartition therefore changes within the candidate
while material and donor identities remain fixed. Exact native captures
confirm each of 28,632 cell masses and the resulting Float32 masses of 7,233
nodes. The nominal, pre-rounding material debits differ by less than 1 mg
from the conserved compiler ledger.

## Native result on SSH Numi

Apple M6, macOS 27.0 (26A425), headless native Metal launch.

- A four-step smoke run completed normally. The longer launch accepted 128
  ordered steps, reaching 2.000 ms. Its largest mechanical-domain monitor
  overrun was root 84 at 1.493e-5 against 1.250e-5. No vascular monitor reported
  an overrun.
- All 28,632 terminal cells remained positive. Minimum terminal/reference
  determinant ratio was 0.94887. The minimum normalized emitted startup
  determinant was 2.6468e-5 against the existing 1e-5 preparation threshold.
- Exact start/end shell audits report zero self- and cross-layer pairs.
  The repaired deep-fat patch reports zero pairs against the posed left
  iliac bank at both captures. These are endpoint checks.
- Dynamic pleural pressure varied from -490.332581 to -490.332489 Pa.
  The terminal audit includes all 34 registered ligament rows. Their net
  force residual was zero and net moment residual was 1.27e-8 N·m.
- Terminal bed reaction was 0.435 N across three rows, against about
  953 N of body weight. The retained configuration is a release
  transient. Supported equilibrium and the later shoulder failure remain
  unqualified.
- Headless throughput was 0.00002026× for the 127 measured steps after the
  first accepted step. The physical prefix and entire terminal node
  capture survived a disk-full failure in the later full tetrahedron text
  dump. The process receipt was lost. Sixteen-second motion and 0.1× native
  presentation remain unqualified.

The existing skin–fat and fascia carries the load in the native scene.
Both source gluteal solids can be tetrahedralized with positive cells.
A separate comparison found 250 pre-existing gluteus-to-skin/fat crossings,
including six at the left bonded interface. The volumetric gluteal candidate
remains an authoring experiment. The 125-inversion and 154-quality-failure
repair concerns the separate full-resolution mesh. This evidence qualifies
the coarse bonded shell only.

## Reproduction

Use the retained `native-inner-fit-022` app and compiler pins, the previous
posterior audit helpers, and the referenced NHTISS4 and bone assets. The
private recovery receipt carries the new fit, mass ledger, launch inputs and
raw compact evidence. The retained comparison launch fixes the nominal time
step, loads, pressure ownership and generated Metal material bundle.

```sh
python3 fit_interface.py "$REFERENCE_RUN" "$PREVIOUS_POSTERIOR_FIT" "$BONE_DUMP" \
  "$TISSUE_READER" "$SHARED_AUDIT_TOOLS" gluteal-registration.json "$FIT_DIRECTORY" --pelvis-bank
python3 "$POSTERIOR_AUDIT_TOOLS/map_mass.py" "$REFERENCE_RUN" "$SHARED_AUDIT_TOOLS" \
  "$FIT_DIRECTORY/posterior-interface-fit.npz" "$FIT_DIRECTORY/mass.json" \
  --solve-transport --mass "$FIT_DIRECTORY/mass.npz"
python3 balance_transfer.py "$REFERENCE_RUN" "$SHARED_AUDIT_TOOLS" "$FIT_DIRECTORY" \
  "$FIT_DIRECTORY/balanced-mass.npz"
python3 native_trial.py build "$BASE_BUILD" "$REFERENCE_RUN" "$FIT_DIRECTORY" "$BUILD_DIRECTORY"
python3 native_trial.py run "$REFERENCE_RUN" "$BUILD_DIRECTORY" "$FOUR_STEP_RUN" --steps 4
python3 native_trial.py run "$REFERENCE_RUN" "$BUILD_DIRECTORY" "$LONGER_RUN" --steps 128 --cache "$FOUR_STEP_RUN"
```

The anatomical assembly combines registered sources. The gluteal and skin
surface data adapt BodyParts3D, Copyright © 2008 The Database Center for Life
Science, licensed under
[CC Attribution-Share Alike 2.1 Japan](https://creativecommons.org/licenses/by-sa/2.1/jp/).
