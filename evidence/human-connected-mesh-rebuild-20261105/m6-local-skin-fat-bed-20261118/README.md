# Load-bearing local skin and fat on the native resting body

This increment makes the existing Metal bed-contact solve compress local skin and fat and transfer the resulting loads to the registered articulated anatomy. It runs in the existing coupled viewer on SSH Numi, Apple M6, macOS 27.0 (26A425). It preserves the mixed-source 72 kg reference adult, MyoSim, tendons, circulation, respiratory feedback, and the released starting pose.

## Physical model

Each of 32 registered skin witnesses represents a patch of the bed-facing surface. One-time projected areas come from the supported rest geometry and weight the compression of a 2.5 mm skin section and 10 mm fat section in series. Effective moduli are 200 and 55 kPa, respectively, with a shared 12 ms retardation time. They are engineering reference values. In-vivo gluteal measurements show material nonlinearity and regional variability. The values used here have no subject-specific or regional calibration. [Isogai et al.](https://doi.org/10.1589/jpts.34.777)

For an area A, the normal compliance is C = h_skin/(E_skin A) + h_fat/(E_fat A). The projected GPU contact solve couples the Kelvin–Voigt correction into the normal row and solves a unilateral impulse. The existing Coulomb friction uses that normal impulse. The existing weighted skin-point Jacobian maps articulated velocity to contact velocity and its transpose delivers the same contact impulse to the skeleton. The 72 kg reference already represents the skin and fat mass.

The contact points are mechanically condensed along the bed normal. In presentation, the skin pass reconstructs the eliminated outer boundary on the rigid bed plane where the accepted underlying geometry compresses a column. Its range is limited by the layer thickness and the existing mesh and skin-to-bed geometry checks remain enabled. A pilot stopped on the skin-to-bed check because it still drew the uncompressed source boundary. The failed receipt is preserved.

## Native motion measurements

| Run | Accepted time | Viewer segment | With process startup | Wall-clock FPS |
| --- | ---: | ---: | ---: | ---: |
| Compliant bed repeat 1 | 16 s | 0.10585× | 0.09858× | 1.66 |
| Compliant bed repeat 2 | 24 s | 0.10568× | 0.10083× | 1.65 |
| Rigid bed control | 8 s | 0.10941× | 0.09513× | 1.71 |

The longest run accepted 12,000 physical steps and audited 376 native frames. The late mean bed reaction was 706.33 N for a weight of 706.32 N. No root assistance was applied. Peak sampled uncompressed-witness overclosure was 5.53 mm during release and at most 3.12 mm during the final two seconds. The latter is the condensed layer shortening. The equivalent force divided by layer stiffness measures up to 0.20 mm of skin and 2.91 mm of fat in that interval. This force-equivalent measure includes damping while a section is moving.

For the matched 6–8 s interval, the compliant body rested 1.36 mm lower than the rigid control at the same support load. The terminal center-of-mass speed was 0.025 mm/s in the longer compliant run. Per sampled step, the sum of external support row impulses divided by the timestep matched the reported support force within 0.000196 N. The motion audit found no invalid rendered triangles and no skin vertices more than 1 mm below the bed at the checked frames. The coupled model completed 4 breaths and 28 modeled filling/ejection cycles, with terminal arterial oxygen and CO₂ partial pressures of 103.15 and 39.09 mmHg.

The model covers localized normal support on a fixed flat bed. Effective compression, fixed tributary areas, and reconstructed contact-side geometry do not resolve in-plane strain, lateral bulging, regionally calibrated pressure, or full skin/fat layer fields. Short native motion does not qualify detailed whole-body tissue contact, anatomical wall-driven cardiac pumping, the five-minute intervention protocol, or the prior 15 fps presentation goal.

## Reproduce on SSH Numi

The tested runtime and inputs are pinned in the launch records. The current built scene can be launched with a fresh run name.

```sh
cd /Users/numi/Work/human-integrated-gaps-20261104/local-bed-layers-002
python3 run.py loaded-repeat --steps 12000
```

Use `--rigid` to select the unchanged hard-contact law and `--steps 4000` to reproduce the shorter matched control. The runner verifies asset and binary hashes before launch. The overlay preserves the dirty Numi Lab checkout and keeps Human-specific source in this repository. To rebuild on the pinned host, run `python3 prepare.py /Users/numi/Work/human-integrated-gaps-20261104/local-bed-layers-rebuild` from this evidence directory. It verifies the six patched sources by SHA-256 and reuses the existing native build. Use `--stage-only` to verify patch application without building. Run `python3 check_motion.py /path/to/native/workdir /path/to/new/evidence loaded-repeat` to check a repeated compliant run.

[Source and binary pins](source-pins.json) · [Motion metrics and raw evidence hashes](metrics.json) · [Native contact patch](local-bed-contact.patch)
