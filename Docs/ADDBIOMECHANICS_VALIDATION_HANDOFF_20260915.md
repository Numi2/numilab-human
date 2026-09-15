# One-subject AddBiomechanics validation handoff

The repository now has a source-bound handoff for the next calibration and
held-out mechanics gate. It intentionally consumes one selected subject
artifact; it does not download the aggregate AddBiomechanics archive or treat
the public dataset as if it were already in this checkout.

The official [AddBiomechanics download page](https://www.addbiomechanics.org/download_data.html)
publishes the high-quality core dataset under CC BY 4.0 and describes measured
ground reaction forces/moments together with estimated kinematics and joint
torques. The complete aggregate is hundreds of gigabytes, so a release must
pin one subject-level artifact and its SHA-256 before fitting or evaluation.

`schemas/human-addbiomechanics-validation.v1.schema.json` defines the manifest.
The compiler requires:

- exactly one recorded adult male with age, height and mass;
- an official AddBiomechanics v1.0 identity, license and hashed subject
  artifact (or an explicit `fixture_only` provenance for tests);
- disjoint calibration and held-out validation trials, with the prediction fit
  scope equal to calibration IDs;
- reference and prediction tables on the same time grid, without resampling;
- declared joint-angle, GRF and joint-moment channels with SI units and
  preregistered thresholds; and
- a prediction source revision and the canonical 12.5 microsecond clock.

Run it with:

```sh
numi human-addbiomechanics-validation --manifest /path/to/manifest.json \
  --output /path/to/validation-receipt.json
```

The receipt reports per-trial and per-split residuals, activity coverage and
the declared clock. It keeps `native_mechanics_qualified`,
`activation_calibration_qualified`, `anatomical_support_loading_qualified`,
`materials_qualified`, `standing_or_walking_qualified` and `release_qualified`
false. This handoff therefore closes an engineering interface for
`calibration.acquire`; it does not claim that a real subject artifact, native
force convergence, anatomical loading, activation calibration, blood transfer,
materials or standing/walking evidence has been supplied.
