# First-contact registration and contact on Apple M6

The full-volume first-contact scene retains 16 consecutive accepted steps at 15.625 microseconds. The step-16 body window took 180.53 seconds for 0.25 milliseconds of simulated time. Its final bed reaction was 0.00186 N against 952.86 N of weight. The run has no complete heartbeat or breath. Supported equilibrium, variable-pressure cycling and 0.1× motion qualification remain open.

## Starting anatomy

The native initial frame matched all 86 initial registered body transforms in the accepted receipt. The other rigid transforms came from the same 157-body native export. The emitted outer skin matched the registered physical boundary. The two stages were checked using exact Float32 triangle predicates across 878 anatomical banks.

At release, 10,764 triangle pairs crossed through one another across 76 muscle, tendon and vessel banks and 2,713 distinct skin triangles. At step 16 there were 10,764 pairs, with 10,761 identical, three new and three cleared. The initial and final count alone conceals those changes. The two brachioradialis banks and two extensor digitorum banks accounted for 2,850 initial pairs.

| Accepted forearm at 0.25 ms | Force-route entries outside skin | Deepest outside route site | Deepest outside muscle vertex |
| --- | ---: | ---: | ---: |
| Right brachioradialis | 0 of 3 | — | 2.44 mm |
| Left brachioradialis | 0 of 3 | — | 2.57 mm |
| Right extensor digitorum | 7 of 36 | 3.08 mm | 13.56 mm |
| Left extensor digitorum | 9 of 36 | 2.85 mm | 14.43 mm |

The extensor entries include the insertion of the left index-finger extensor, 0.89 mm outside. The executed route and the surface representation have separate owners. Refit the brachioradialis bulk using the enclosed named sites as constraints. For the extensors, reconcile skin, the phalanges, attachment sites and the envelope in one registered frame before fitting the bulk. Preserve sourced attachment identities and verify the resulting route lengths and moment arms across joint motion. A rigid muscle-surface edit cannot change an exterior force site.

## Native contact trials

Twenty-two of the 29 deep-fat intersection pairs at the accepted endpoint share exactly one vertex. They overlap beyond that vertex. The legacy candidate rule excludes such triangle pairs. Five matched experimental variants retained the original capacity, constitutive laws, convergence tolerances and independent physical acceptance checks.

| Variant on the complete scene | Root-zero candidates | Root-zero active contacts | Accepted steps |
| --- | ---: | ---: | ---: |
| First-contact control | 3,635,156 | 658,852 | 16 |
| Independent features of one-vertex fans | 6,024,454 required | Capacity exceeded | 0 |
| Swept angular separation | 4,283,427 required | Capacity exceeded | 0 |
| Include swept triangle-plane separation | 2,115,962 | 1,115,862 | 0 |
| Fit the barrier width to the immutable disjoint features | 2,115,920 | 1,115,862 | 0 |
| Use the same width for collision time and force | 2,115,781 | 659,089 | 0 |
| Triangle-plane separation with the original adjacency policy | 1,467,679 | 658,852 | 16 |

Counts for a completed root are the final workspace observations in that root. The two capacity rejections record requested slots. The width change reduced the measured nonvascular residual from 3.02117 to 0.007349. Using that width during collision detection removed most of the excess active rows, with a remaining residual of 0.009700 against a physical threshold of 0.0000125. The scene rejected both changes at the first step. The unchanged-adjacency speed control took 191.50 seconds in its body window. This single comparison measured a slowdown despite a 60% reduction in candidate count.

Keep the original full-volume control as the reference. The new shaders remain experiments. The full-scene checks still find 29 deep-fat crossings and eight distinct fat cells failing shape, volume or stretch at step 16. Separate work is needed to resolve registration, mesh conditioning and static bed-load transfer before sustained physiological qualification.

The GPU fixtures checked 22 shared-vertex witnesses, all 29 deep-fat intersection pairs, 256 triangle-plane cases and 16 feature-width cases. The release archive includes shader sources, compiled libraries, test programs, launch configurations, failure traces and exact registration witnesses. The recovery manifest records the independently downloaded archive and 104 verified member hashes.

On Numi, launch the native release with `~/Work/run-numi-human-rest --full-first-contact <unused-output-directory>`. Check a completed run with `~/Work/run-numi-human-rest --check-full-first-contact <run-directory>`. The second command checks the terminal state and returns a failure when anatomy or material checks fail.
