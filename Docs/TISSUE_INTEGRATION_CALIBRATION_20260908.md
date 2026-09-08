# Anatomical tissue integration and calibration, 8 September 2026

This increment preserves articular cartilage material provenance in the native
payload, implements the source ligament fibre law in Matter, corrects the
costal material parameterization, and adds a reproducible raw-data calibration
candidate. It does **not** qualify prestressed whole-joint or whole-human tissue
mechanics.

## Source law and anatomy

`NHKNEE1` ABI 3 retains the solid Mooney–Rivlin parameters for all four source
cartilage regions: `FMC`, `PTC`, `TBC-L`, `TBC-M`. Previously these values
appeared in the manifest but were zero in device records. Each now carries
`c1=2.54 MPa`, `c2=0`, `K=100 MPa`, and material flag 4. The native decoder
accepts historical ABI 2 separately and rejects missing or changed ABI 3
cartilage parameters. No fit overwrites the source model.

The fresh left payload is `2e38201528de25911ea496164602ea7e823cd9c2e3c94efa550ee52689324ae5`.
Its geometry, topology and attachment bytes are identical to the prior
`c5116b…` payload; only ABI and four material records change. Named live body
bindings remain femur 145, tibia 150 and patella 156. This run regenerated and
verified the left payload; mirrored-right ABI 3 qualification remains pending.

Matter now provides `fiber_exp_linear(stretch,c3,c4,c5,lambda_max)`. It evaluates
the source tension threshold, exponential toe and linear tail, with
`q = stretch*dW/dstretch`, rather than treating `q` as the energy derivative.
It retains the actual tangent jumps. The full material composes the admitted
isochoric prestrain with both fibre and matrix deformation. All six admitted
ligament/tendon sources have `c2=0`; the probe rejects nonzero `c2`.

The implementation follows [FEBio's source law](https://github.com/febiosoftware/FEBio/blob/80868f1cba7408eac6fb8fec1c08654a4ec5b12f/FEBioMech/FEUncoupledFiberExpLinear.cpp).
Energy uses 16-point Gauss–Legendre integration within an explicit bounded toe
domain; stress and tangent are analytic. Coefficients must be material inputs,
and deformation/state-dependent shape coefficients fail admission. The source
law is an explicit `--source-fiber-law` probe option; the existing live reduced
fibre owner and its force ownership remain unchanged.

On Apple M4 Pro with Metal validation:

- All six material laws pass FP64 energy/stress/tangent comparisons and 3,060
  Metal scalar checks, including threshold/transition neighbours.
- PCL's 3,714 nodes and 14,379 tetrahedra pass the submicron prescribed-bone
  reaction transaction, bitwise replay and rollback. Maximum displacement is
  `2.31598e-7 m`, with `J` in `[0.999849,1.00015]`. Production owner fraction is 0.
- LCL source prestress, eight ramp stages plus eight settle steps and rate
  exponent 2, rejects stage 1 with status 9, index 11, four microsteps and
  16 FGMRES iterations. Neither equilibrium nor loaded flexion is qualified.
- Missing optional-buffer bindings exposed by Metal validation were corrected
  in both provisional validation and reaction assembly. Count-zero inputs use
  typed inert records; they create no physical contact or force source.

## Costal correction and remaining binding work

The v1 energy's logarithmic coefficient was named and set as physical bulk
modulus, although the energy uses Lamé lambda. The v2 law uses
`mu=7.5862069 MPa`, `lambda=68.2758621 MPa`, giving `K=73.3333333 MPa`,
`E=22 MPa` and assumed `nu=0.45`. The actual compiled symbolic tangent passes
all 81 infinitesimal components and stress finite differences. These are still
population starting values, not experimentally calibrated costal tissue.

The unchanged 14-region cartilage geometry passes its prescribed-displacement
probe with Metal validation, replay and rollback. This does not create live
rib articulation. All source rib and sternal attachments resolve to torso 20.
NHCART1 stores atlas coordinates without a live registration or inertia
partition. Independent rib/sternal fits would introduce 20.9–51.8 mm of
unrecorded initial deformation. A coherent registered reference volume,
explicit cooked tissue/rigid mass partition, and source thorax articulation
are still required before promoting breathing mechanics. No mesh-derived
joint or duplicate tissue mass was introduced.

## Experimental cartilage candidate

The new `tissue-calibration` command imports the pinned, CC-BY-4.0
[Open Knee cartilage archive](http://archive.simtk.org/oks/tissue/oks003/cartilage/),
dataset [10.18735/WTJZ-N328](https://doi.org/10.18735/WTJZ-N328), associated with
[Chokhandre and Erdemir's study](https://doi.org/10.1016/j.jmbbm.2020.104025).
It checks exact sizes and SHA-256 hashes for three Mach1 histories, three OTMS
thickness records, the protocol README and license. Raw force is **gram-force**,
converted with `0.00980665 N/gf`; position is millimetres.

One patellar plug, `oks003-PTC-MCXX-01`, has repeated unconfined tests. Repeats
01 and 03 train a single apparent shear coefficient; repeat 05 is excluded
from fitting. This is a test-day split within one plug, not a specimen or donor
holdout. Contact reference comes from the recorded Move Absolute command plus
the protocol's 0.300 mm offset and is checked against the measured contact
position. The three holds must match the intended 5%, 10%, 15% strains.

The fitted compressible Neo-Hookean response assumes `nu=0.45`, zero lateral
traction, a 5 mm plug, no platen friction, and a preloaded reference proxy.
It fits the time-weighted last 300 seconds of each 1,800-second hold. Force
continues to relax during the final segment; equilibrium is not assumed.

| Quantity | Result |
|---|---:|
| Apparent shear coefficient | 70,244.4775 Pa |
| Training points | 6 |
| Training force RMS error | 0.036642 N, 7.37% of measured RMS |
| Held-out repeat points | 3 |
| Held-out force RMS error | 0.031281 N, 5.96% of measured RMS |
| Held-out maximum absolute error | 0.049058 N |

Bulk modulus, density, friction, stress-free reference, anisotropy,
permeability and relaxation spectrum are not identified by this fit. The
emitted `.nmatter` is an **unqualified candidate**. Its native compile/package
roundtrip is a format check, not a specimen boundary-value validation. It is
not assigned to the whole patella, other regions, other donors or the mirrored
knee. The source model is solid Mooney–Rivlin; it supplies no biphasic or
permeability parameters.

During review, a preliminary importer incorrectly used a new force crossing
in the fast ramp as the reference, yielding approximately 8%, 13%, 18% strains.
That candidate was rejected. The documented protocol corrected this before
publication; a dedicated test prevents recurrence. Because the held-out
responses were already visible during that correction, this receipt remains
a development candidate, not a blinded physical validation certificate.

## Reproduce

```sh
PYTHONPATH=src python3 -m numilab_human.cli tissue-calibration \
  --sources Build/oks003-cartilage-raw --fetch \
  --output Build/oks003-cartilage-candidate

PYTHONPATH=src python3 -m numilab_human.cli open-knee-oks003-payload \
  --sources Sources --open-knee Sources/open-knee-oks003 \
  --registration Build/fullbody-articular-v3.registration.json \
  --python .venv-myosim/bin/python --side left --output Build/knee-material-v4

# In the native Numi Lab build:
numi-matter-fiber-check --cpu-only
MTL_DEBUG_LAYER=1 numi-matter-fiber-check
metalrobo_numilab_human_open_knee_ligament_probe \
  open-knee-oks003-left.nhknee --payload-only
MTL_DEBUG_LAYER=1 metalrobo_numilab_human_open_knee_ligament_probe \
  open-knee-oks003-left.nhknee --source-fiber-law --tissue=PCL
```

The [evidence directory](media/tissue-integration-20260908/) preserves exact
source/candidate identities, native logs and the failing prestress result.
Next promotion requires a prestress equilibrium solve and a preload-consistent
specimen boundary-value study with mesh/timestep convergence and a newly
sealed validation cohort. Live anatomical integration additionally needs the
mass partition and nonduplicated same-root force ownership gates.
