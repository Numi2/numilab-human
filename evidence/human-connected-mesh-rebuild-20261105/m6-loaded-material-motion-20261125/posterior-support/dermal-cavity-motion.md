# Tissue remeshing under the integrated native release

SSH Numi ran the full current body with the registered gravitational and bed loads, muscle and ligament forces, variable pleural pressure and accepted-state transactions. Each completed trial advanced two 7.8125 µs steps. The physical question was whether remeshing the most sensitive dermal and adipose cells could improve the moving tissue while retaining the measured mechanical response.

| Native endpoint | Original control | Connected pelvic repair and two cavities | Original pelvic pose and carrier-routed cavities |
| --- | ---: | ---: | ---: |
| Inverted cells | 0 | 0 | 0 |
| Cells below shape limit | 40 | 40 | 40 |
| Cells outside volume-ratio limit | 7 | 7 | 7 |
| Cells outside principal-stretch limits | 506 | 499 | 502 |
| Crossing wall and cortical triangle pairs | 6568 | 6568 | 6568 |
| Pelvic overlap, mL | 2.91540 | 2.89191 | 2.91540 |
| Dermal nodal force field change, % | 0.000 | 0.227 | 0.065 |
| Adipose nodal force field change, % | 0.000 | 10.511 | 0.031 |
| Bed reaction, N | 10.992 | 10.985 | 10.989 |
| Body runtime, s | 85.62 | 95.79 | 85.84 |
| Launch-to-exit, s | 233.06 | 247.16 | 234.98 |

The force comparison sums the native constitutive force contributions at each corresponding material node, then compares the Euclidean norm per tissue. All registered interface contact laws remain active during motion. Forces here measure the elastic response of each tissue.

Seven principal-stretch failures cleared in the two-cavity trial. All common and replacement cells passed the comparison for newly introduced failures. Its adipose force field changed by 10.51%, exceeding the 1% comparison gate. Opening registration and topology were therefore tested separately. The batch searched 533 failed cells. Seven disjoint cavities cleared reference and matched-pose checks. Two required fitting their assigned masses to the measured source quadrature and testing those flows with the native router.

The original-pose trial introduced 1 failure memberships on common cells and 0 on replacement cells. The exact crossing comparison found 0 added and 0 removed identities.

The joint-stops control and the experimental profiles represent the opening gravity release. Their measured support falls far below the 953 N gravitational load. The complete scene still requires a supported opening pose and a load path through the posterior soft tissue into the skeleton that sustains the body as the contact footprint grows. These short runs do not qualify a breathing cycle, 16 seconds of motion or 0.1× throughput.

Replay the pelvic diagnostic on SSH Numi with `bash scripts/run_resting_human_on_numi.sh --dense --posterior-support pelvic-material --steps 2`.

Replay the original-pose material trial with `bash scripts/run_resting_human_on_numi.sh --dense --posterior-support conservative-cells --steps 2`. This seven-cavity profile remains experimental.

## Isolating the neighboring strain failure

Cell 456199 in dermis was 0.36 mm from the nearest changed cavity. Its reference altitude was 0.83 µm. Its minimum principal stretch changed from 0.692 to 0.639 with 0.51 µm of endpoint movement. The repeat kept the original registered material pose and the four source-routed cavities at least 49 mm away.

The repeat accepted 15.625 µs in 85.49 s of body runtime. Across the skin and fat, 0 cells inverted, 40 fell below the shape limit, 7 exceeded the volume-ratio bounds and 504 exceeded principal-stretch bounds. There were 0 newly failing comparisons on common cells and 0 on replacement cells. Across 185 cortical banks, 0 crossing pairs appeared and 0 disappeared. The bed carried 10.992 N.

Against the native control, dermal nodal force changed by 0.0027% and adipose by 0.0185%. The first captured load has all 128 generalized coordinates. The next capture records the three translational root forces. 0 of those comparisons exceeded tolerance. Of the 22 coupled respiratory, cardiac, blood and gas outputs compared, 0 differed at the two matched times.

The four-cavity release passes the short material, body-load and local crossing comparisons. The native launcher now selects this revision for the two-step release. The previous control remains available with `--dense --posterior-support joint-stops`. Longer motion, support equilibrium and sustained performance remain open.

## Posterior support decision

The accepted skeleton receives 1.44 N through the measured skin–bone interface. Its force balance requires 811.9 N if the other measured forces stay fixed. The retained bilateral iliocostalis pressure-only tests on their registered surfaces carry at most 0.20 N when scaled to the observed 6.04 kPa peak. This bound applies to those two fixed muscle surfaces. The next mechanical change needs to recruit a substantially broader posterior contact area and transmit its reaction through connected tissue into the skeleton. Longer releases and finer versions of those two passive surfaces do not establish that capacity.

The [recovery manifest](./dermal-cavity-recovery.json) records the independent GitHub download and the SHA-256 of all 376 restored files, including the native executables, source-mass allocations, accepted geometry and traces.
