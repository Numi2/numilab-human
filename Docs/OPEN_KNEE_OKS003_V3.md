# Open Knee(s) oks003 source-material payload v3

This revision upgrades `NHKNEE1` from geometry/topology-only material
provenance to an executable source-directed tissue boundary. ABI 2 stores the
exact homogeneous fibre axis and `c1`, `c2`, `c3`, `c4`, `c5`, `lam_max`,
bulk-modulus, and in-situ-stretch values for `ACL`, `PCL`, `MCL`, `LCL`, `PTL`,
and `QAT` in every region record.

The source gate covers all four files that own this result:

- `Geometry.feb`: `3642bd368bbc867569f181fa76129f746470e807e3977585d3803f092dd11262`
- `ModelProperties.xml`: `0ac446ce098b9a09505992eb4f4419c7b944cd57a4afbf6392b137f4806603c1`
- `FeBio_custom.feb`: `00b6efb53ad7e7330296cbb9569d358d48ed60819e22732e6149db6fb98a158a`
- `license.txt`: `d72918838b4adf30979d2a26c23837f0ca05185ba799a3a4fe1fe1b4c05b20b8`

## Reproducible artifacts

- left: `c5116b2cff086dae43e71a282ca5a15088b682875af4f0c0c1beb563fa255b4e`
- mirrored right: `27efaed583d3a0dcae5b27d1481c336f9d630a6cce0bd64420577b06170c56be`
- bytes per payload: `34357400`

Two independent left compilations were byte-identical. The right output keeps
the previously qualified connectivity-parity correction and mirrors each
registered fibre axis through the same sagittal plane; fibre-axis sign remains
physically equivalent, but the explicit mirrored vector is retained.

## Build

```sh
numilab-human open-knee-oks003-payload \
  --sources Sources \
  --open-knee Sources/open-knee-oks003 \
  --registration Build/fullbody-articular-v3.registration.json \
  --output Build/open-knee-transiso-v3-left \
  --side left

numilab-human open-knee-oks003-payload \
  --sources Sources \
  --open-knee Sources/open-knee-oks003 \
  --registration Build/fullbody-articular-v3.registration.json \
  --output Build/open-knee-transiso-v3-right \
  --side right
```

## Evidence boundary

ABI 2 is exact source admission. The Apple Matter runtime consumes the source
fibre direction through a smooth exponential tension law, but that runtime law
is not yet the exact FEBio piecewise exponential-linear/`Ei` formulation.
Applying the final ACL/MCL/LCL in-situ stretches as a one-step jump failed the
bounded nonlinear/performance gate. The values remain admitted and fail-closed;
the current passing mechanics result uses neutral stretch until a staged
prestress/equilibrium ramp is implemented and qualified.
