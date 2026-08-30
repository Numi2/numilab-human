# Exact costal-cartilage mechanics v1

This increment turns the fourteen named BodyParts3D 4.0 costal-cartilage
surfaces into an executable, source-locked FEM payload. It is the first
cartilage mechanics slice in NumiLab Human; it does not claim that articular
cartilage, menisci, ligaments, or whole-thorax coupling are complete.

## Source and topology receipt

The compiler reads the exact `part_of` OBJ members below from the pinned
`partof_BP3D_4.0_obj_99.zip` archive (SHA-256
`9fbc713fffeee924a5a657d9813d84d7eb957bded63adb854931dd5e3eb61c97`,
CC BY 4.0):

| Side | Ribs 1--7 BodyParts3D members |
| --- | --- |
| left | `FJ3239`, `FJ3242`, `FJ3245`, `FJ3248`, `FJ3251`, `FJ3254`, `FJ3255` |
| right | `FJ3333`, `FJ3335`, `FJ3337`, `FJ3339`, `FJ3341`, `FJ3343`, `FJ3345` |

Exact neighboring rib surfaces and the sternum/manubrium surfaces classify
non-overlapping 4 mm rib and sternal attachment bands. Deterministic voxel
connectivity fills each closed source surface independently; no cartilage is
inferred between ribs 8--12 and no knee, TFCC, meniscus, or ligament geometry
is invented.

The deterministic `NHCART1` payload receipt is:

| Quantity | Value |
| --- | ---: |
| named bilateral regions | 14 |
| FEM nodes | 13,516 |
| positive tetrahedra | 46,278 |
| attachment nodes | 2,871 |
| rest volume | 0.000101038375 m3 |
| mass at 1,100 kg/m3 | 0.111142213 kg |
| maximum source-volume error | 1.86362% |
| payload bytes | 1,304,976 |
| payload SHA-256 | `f5c58b4ddf8a97f631fd21aa7c861bbfa57bc8f863ff20a04165be4db0b5e5cc` |

The runtime decoder fails closed on the archive identity, exact ordered member
IDs, bilateral rib levels, region ranges, attachment counts, finite node mass,
node ownership, positive tetrahedral volume, and trailing bytes.

## Material starting point

The v1 law is a compressible Neo-Hookean pseudo-elastic starting model with
`E = 22 MPa`, assumed `nu = 0.45`, derived shear modulus `7.586 MPa`, derived
bulk modulus `73.333 MPa`, and `25 Pa s` numerical viscosity. The modulus comes
from Forman et al.'s whole-segment cadaver experiments, which report
`22 +/- 13.6 MPa` and a `4.8--49 MPa` range. Weber et al. independently show
that human costal cartilage is anisotropic and age-dependent:

- Forman et al. (2010), DOI
  [10.1080/15389588.2010.517254](https://pubmed.ncbi.nlm.nih.gov/21128192/)
- Weber et al. (2021), DOI
  [10.1038/s41598-021-93176-x](https://www.nature.com/articles/s41598-021-93176-x)

Therefore homogeneous isotropy, near-incompressibility, density, and viscosity
are explicit v1 assumptions. This is not subject-, age-, calcification-, rate-,
injury-, or clinical-qualified material behavior.

## Apple Metal execution gate

`metalrobo_numilab_human_costal_cartilage_probe` decoded the exact payload and
ran all fourteen disconnected anatomical regions in one Numi Matter FEM world
on Apple M4 Pro. Each region received a 1 N rib-band structural load while its
source-classified sternal band was kinematically attached. External loads,
kinematic targets, nonlinear FEM, fixed-node reactions, Human acceptance, and
rollback shared one borrowed Metal command buffer.

| Gate | Apple M4 Pro result |
| --- | ---: |
| source regions with deformation | 14 / 14 |
| fixed sternal nodes with region coverage | 1,582 |
| maximum displacement | 2.70489 um |
| minimum / maximum `J` | 0.999929 / 1.00012 |
| fixed-anchor reaction L1 | 0.00385917 N |
| accepted replay | bitwise identical |
| rejected transaction | rollback verified |
| wall time | 42.51 s |

The small reaction after one `0.1 ms` step is a transient structural witness,
not a static force-balance calibration. The qualification probe temporarily
replaces 10% of its synthetic paired terminal load so the existing two-way
assembly is exercised. The production cartilage owner fraction remains zero:
the source bands must next bind to their named live articulated rib and
sternal bodies, return equal-and-opposite accepted reactions, and demonstrate
non-duplicated load ownership over a breathing/load cycle.

The preserved device transcript is
[`Docs/media/numi-human-costal-cartilage-v1/costal-cartilage-m4-pro.transcript.txt`](media/numi-human-costal-cartilage-v1/costal-cartilage-m4-pro.transcript.txt).

## Remaining completion boundary

This closes neither the broader cartilage row nor the user's visible anatomy
review. The next useful mechanics gates are:

1. bind all fourteen bands to named live ribs/sternum and validate loaded
   multi-pose thorax geometry from front, side, oblique, and rear views;
2. add subject/age/calcification parameter fields and held-out deformation
   calibration before stronger physical claims;
3. source superior free knee/meniscus, shoulder, wrist/TFCC, and spinal-disc
   geometry before implementing compliant joint contact;
4. add named ligament paths and nonlinear tensile laws without replacing joint
   constraints until each new owner passes force and replay gates.
