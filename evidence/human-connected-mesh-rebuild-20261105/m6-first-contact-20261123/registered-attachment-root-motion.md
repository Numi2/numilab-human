# Registered attachments at the start of coupled motion

The full native Human previously showed a large reversing load on the skeleton when the first physical interval was divided in two. At the same joint coordinates, the scene's standalone initialization used ordinary floating-point forward kinematics while the running articulated operator used compensated root translation. The fixed tissue points therefore started at different positions from the targets supplied to the first live solve. [Earlier attachment measurements](deep-fat-native-motion.md)

The corrected launcher evaluates all 270 registered material attachment targets with the compensated articulated operator and the production Metal target kernel. It installs their positions through Matter's validated initial-state restore before time advances. Maximum change to an initial prescribed point was 0.541 µm. The node masses, reference coordinates, constitutive properties, initial velocities and anatomical body-local attachment coordinates remain those of the compiled model. The first implementation attempted to encode this displacement as a newly authored material reference. The compiler correctly rejected an identified material. The tested implementation changes only the initial state. [Rejected trial and interrupted export recovery](attachment-release-interrupted-recovery.json)

## Matched physical interval

SSH Numi, Apple M6, macOS 27.0, arm64. Each case ends at 15.625 µs. Each subdivision uses the same material registration, articulated configuration, flat bed, variable pleural coupling and zero root assistance. The two-half-step control used the trial executable with the initialization disabled.

| Projected tissue force on skeletal root | Original, N | Corrected, N |
| --- | ---: | ---: |
| One full step, magnitude | 23.157 | 2.071 |
| First half step, magnitude | 91.548 | 2.057 |
| Second half step, magnitude | 91.055 | 2.008 |
| Vector difference between full step and first half step | 68.461 | 0.053 |
| Vector difference between full step and the half-step mean | 22.818 | 0.066 |

The original half-step directions have a cosine of −0.99903. The corrected directions have a cosine of +0.999995. Both corrected runs completed through the ordinary coupled owner. A previous two-step export stopped at its disk reserve after both roots were accepted. Repeating the complete native run reproduced the accepted root, body, respiration and frame identities. [Numerical comparison and timings](registered-attachment-root-comparison.json) · [Recovered executable, original commands and physical captures](registered-attachment-root-recovery.json)

## Physical qualification

At the two-step endpoint the bed carried 0.294 N against 952.864 N of weight. The independent audit found 10,132 nonocular skin-to-tissue triangle intersections across the 878 banks examined, zero bone crossings, zero skin self-intersections and zero registered layer crossings. Twelve distinct fat cells failed the existing motion checks. Eleven exceeded the stretch envelope and one fell below the shape floor. Peak stretch was 2.752. These results establish the correction to the startup force and leave supported rest and anatomical motion qualification open. [Whole-body endpoint](registered-attachment-two-step-endpoint.json) · [Individual material cells](registered-attachment-two-step-cells.json)

The complete two-root process took 189.65 s, including preparation and the terminal frame export. Its end-to-end factor was 8.24e−8×. The body window measured 6.66e−7×. This interval is too short for a heartbeat or breath. The 16-second and longer criteria remain open.

The startup impulse and the forearm distortion have separate remedies. Correcting the attachment state reduced the force substantially at both timestep sizes. At each timestep the matched original and corrected runs agreed in bed reaction and differed in maximum skin/fat stretch by less than 0.000003. The next physical test should trace the loads on the failing forearm cells and establish a supported equilibrium with measured vertical reaction and moment balance.
