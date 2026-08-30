# Articular placement v1

## Outcome

The bilateral BodyParts3D long bones are now registered against the pinned
MyoSim/OpenSim mechanics with a proper, bounded, uniform similarity fit. This
fixes the placement failure that remained hidden by nearest-surface continuity
checks: two adjacent bones could have a small gap at one vertex while the
actual head, socket, or joint axis was visibly displaced.

The admitted transform family is deliberately narrow:

- one rotation with positive determinant;
- one translation per rigid source segment;
- one uniform anthropometric scale in the 0.93--1.07 interval for humerus,
  radius, ulna, femur, and tibia;
- no reflection, anisotropic warp, per-vertex deformation, or extra joint;
- foot and five-toe compounds retain their existing rigid ownership and zero
  independent toe articulations.

The fit is selected on source/target surface samples, checked on held-out
vertices, and then rejected unless the articular sphere and mechanics axis
also pass explicit gates. The femoral center refinement is translation-only,
bounded to 6 mm, and cannot alter the admitted rotation or scale.

## Quantitative placement gates

| Segment | Side | Uniform scale | Articular-center residual | Center-to-mechanics-axis | Radius residual |
| --- | --- | ---: | ---: | ---: | ---: |
| humerus | left | 1.045464 | 1.289 mm | 2.610 mm | 0.013 mm |
| humerus | right | 1.047625 | 1.081 mm | 2.794 mm | 0.280 mm |
| femur | left | 0.962462 | 1.202 mm | 2.129 mm | 0.117 mm |
| femur | right | 0.961007 | 1.288 mm | 2.194 mm | 0.376 mm |

The remaining long-bone scales are radius 1.026328/1.030004, ulna
1.004314/1.012055, and tibia 0.967968/0.969512 for left/right respectively.
Every admitted rotation has positive determinant. The cumulative lower-body
continuity maximum is 3.876 mm with no failed transition.

The upper-limb pose audit passes six source poses, 312 continuity evaluations,
and 156 bilateral parity evaluations. Its worst posed interval is 12.596 mm
for left ulna-to-triquetrum against the explicit 13 mm ulnocarpal allowance;
the worst bilateral difference is 1.267 mm for radius-to-lunate against 2 mm.
The machine-readable receipt is the
[upper-limb multi-pose audit](media/numi-human-articular-placement-v1-1024/upper-limb-multi-pose-audit.json).

## Native Apple visual review

These frames were rendered by the clean Release visual probe on Apple M4 Pro.
Bone-only views were reviewed before the soft-tissue overlay so muscle surfaces
could not hide a misplaced joint. Neutral and non-neutral poses were inspected
from front, rear, side, and oblique cameras for both sides.

### Shoulder seating through elevation

<p align="center">
  <img src="media/numi-human-articular-placement-v1-1024/neutral-right-shoulder-front.png" width="24%" alt="Neutral right shoulder, front" />
  <img src="media/numi-human-articular-placement-v1-1024/neutral-left-shoulder-front.png" width="24%" alt="Neutral left shoulder, front" />
  <img src="media/numi-human-articular-placement-v1-1024/raised-right-shoulder-front.png" width="24%" alt="Elevated right shoulder, front" />
  <img src="media/numi-human-articular-placement-v1-1024/raised-right-shoulder-rear.png" width="24%" alt="Elevated right shoulder, rear" />
</p>

The humeral head remains seated relative to the glenoid in the elevated source
pose. This is a kinematic/articular-center result, not a loaded shoulder
stability or cartilage-contact claim.

### Elbow, wrist, and hand chain

<p align="center">
  <img src="media/numi-human-articular-placement-v1-1024/neutral-right-elbow-wrist-oblique.png" width="24%" alt="Neutral right elbow and wrist" />
  <img src="media/numi-human-articular-placement-v1-1024/neutral-left-elbow-wrist-oblique.png" width="24%" alt="Neutral left elbow and wrist" />
  <img src="media/numi-human-articular-placement-v1-1024/flexed-right-elbow-side.png" width="24%" alt="Flexed right elbow" />
  <img src="media/numi-human-articular-placement-v1-1024/right-functional-fist-front.png" width="24%" alt="Right functional fist" />
</p>

### Forward knee flexion and bilateral five-ray feet

<p align="center">
  <img src="media/numi-human-articular-placement-v1-1024/neutral-right-knee-front.png" width="24%" alt="Neutral right knee, patella anterior" />
  <img src="media/numi-human-articular-placement-v1-1024/neutral-left-knee-front.png" width="24%" alt="Neutral left knee, patella anterior" />
  <img src="media/numi-human-articular-placement-v1-1024/flexed-right-knee-side.png" width="24%" alt="Right knee flexed forward" />
  <img src="media/numi-human-articular-placement-v1-1024/flexed-left-knee-side.png" width="24%" alt="Left knee flexed forward" />
</p>

<p align="center">
  <img src="media/numi-human-articular-placement-v1-1024/neutral-right-foot-oblique.png" width="40%" alt="Right five-ray foot with medial hallux" />
  <img src="media/numi-human-articular-placement-v1-1024/neutral-left-foot-oblique.png" width="40%" alt="Left five-ray foot with medial hallux" />
</p>

Both patellae remain anterior and the knees flex forward in the exercised
source coordinates. Each foot retains five distinct rays; the hallux is the
medial first ray and is not independently articulated.

## Tendon-to-bone compatibility

The newly registered bones were recompiled with the qualified compliant
muscle payload, equality program, and NHTENDON3 transfer laws. The M4 Pro
reference probe executes 416 muscles and all 832 endpoints: 638 distributed
envelopes, 194 exact source-point fallbacks, and all 18 named foot/hallux
migrations. Maximum endpoint migration remains 17.262 mm and maximum reference
path recalibration is 10.549 mm. Replay is byte-identical.

CPU force/moment residuals are 0.0000913 N and 0.00000712 N m. Metal
force/moment residuals are 0.000123 N and 0.00000813 N m, with 0.000211 N
nodal transfer parity error. The paired qualified muscle artifact is required;
an older pre-architecture NHMYO payload correctly fails reference calibration
because it contains zero compliant architecture values.

The global 638/194 disposition split is unchanged, but 16 individual
endpoints changed disposition under the same fail-closed conditioning gates:
eight newly admit distributed envelopes and eight revert to their exact
source-authored point law. Nothing is silently forced onto a poorly
conditioned surface patch.

<p align="center">
  <img src="media/numi-human-articular-placement-v1-1024/qualified-right-shoulder-overlay-front.png" width="40%" alt="Qualified shoulder geometry with source routes and transfer envelopes" />
  <img src="media/numi-human-articular-placement-v1-1024/qualified-right-foot-overlay-oblique.png" width="40%" alt="Qualified foot geometry with source routes and transfer envelopes" />
</p>

The cyan route centrelines are source biomechanics paths, and the terminal
envelopes/point laws are force-transfer diagnostics. They are not a rendered
collagen tendon volume. Dense whole-model overlays are therefore evidence of
route and endpoint execution, not anatomical beauty renders.

## Evidence boundary

This establishes bounded source/mechanics bone registration, explicit
articular-center and joint-axis agreement, bilateral posed continuity, correct
source-coordinate knee direction, five-ray rigid-foot preservation, and
execution of the paired tendon-to-bone transfer program on Apple GPU. It does
not establish clinical subject registration, cartilage contact under load,
ligament-constrained stability, deformable tendon/fascia continuum behavior,
skin contact, balance, or gait.
