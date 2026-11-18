# Passive hand mechanics and cardiac pressure trials on Apple M6

The matching native source is the retained integrated Human probe over runtime revision
`f75955f08d6edf0f85027eac7614017ac9fbc996`, with the prior regional bed and thoracic trial overlays.
Launch and build receipts give exact inputs and the binary hashes. Native force calculations,
circulation, Brain delivery, gas exchange, collision checks, and accepted-state publication
use the existing owners. The scene uses the condensed bed-contact skin/fat model.

## Hand mechanics

The existing wrist and four-finger passive program was enabled first. It reduced four-second
skin self-intersection pairs from 610 to 455. Then eight source-validated thumb rows were
added to the same passive force and stiffness program. The elastic terms reference the
registered neutral pose. CMC axes use 0.15 Nm/rad each, MCP uses 0.10 Nm/rad, and IP uses
0.05 Nm/rad. These are reference model assumptions, supported by the
[published thumb model](https://pmc.ncbi.nlm.nih.gov/articles/PMC2804669/).
The bilateral wrist and finger source has no geometry, source-weight, mass, body-lock, or joint-limit edits.

| Exact native 4 s capture | Skin self-pairs | Skin–bone | Skin–muscle | Skin–vessel | Total skin to target |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regional bed control | 610 | 3,356 | 10,533 | 322 | 14,211 |
| Existing passive wrist and fingers | 455 | 3,147 | 10,105 | 260 | 13,512 |
| Added passive thumb | 39 | 2,292 | 9,271 | 270 | 11,833 |

These are exact Float32 intersecting triangle-pair counts at matched physical time. They
do not measure penetration depth, continuously sampled clearance, or physical contact
between the embedded visual layers. Some individual banks regress. Of the 39 self-pairs,
36 are in arm-to-torso attachment blends. Three are near the left thumb and index web.
The same outer reference mesh and source influence bytes were used in all three runs.
The staged dual-quaternion diagnostic increased self-pairs to 643 and was not adopted.

`thumb-passive-reference.patch` applies to the retained prior native app source. Enable it
with `--persistent-source-passive-joint-tissue` and
`NUMI_HUMAN_RESTING_THUMB_PASSIVE_REFERENCE=1`. The compiled trial additionally contains the
opt-in pressure experiments below, which were disabled for isolated hand qualification.
The capture I/O patch clones already verified load-time map records where possible.

## Cardiac pressure admission

Immediate absolute gauge pleural loading rejected step 222 after 0.444 s. Five seconds of
smooth initialization from the authored circuit pressure continued onto the unscaled
pleural boundary and rejected step 2779 after 5.558 s. Both first failures returned
common-field status 5, which indicates a successful volume solve outside all certified
coordinate boxes. Rejected candidates were withheld by the existing transaction.

Numerical minimax searches across all 125 retained boxes returned relative residuals of
approximately 0.000907 and 0.000471. The prescribed relative volume tolerance is 0.000020.
Those searches characterize the retained model and do not prove that the anatomy at an
uncertified candidate coordinate is invalid. A source-verified cardiac certificate for the
updated pressure trajectory remains required. Neither pressure trial is promoted.

## Native motion and timing

The passive-thumb reference with the authored constant vascular exterior pressure accepted
all 8,000 steps over 16 simulated seconds. Sustained throughput was 0.10665× including
native viewer cadence, accepted-state observation, Brain, physiology, and body mechanics.
The viewer rendered 251 native frames, approximately 1.7 per wall-clock second.
The whole process took 161.05 s including load, initialization, and finish (0.09935×). An earlier
attempt stopped at the disk guard after 8.896 simulated seconds. Verified lossless capture
compression provided space for the completed rerun.

The accepted record has three completed breaths and 19 LV filling/ejection cycles, a final
stroke volume of 70.05 mL and a maximum sampled blood-volume discrepancy of 0.0070 mL.
Mean final-two-second bed reaction was 706.28 N against 706.32 N weight, with zero root
assistance. All 251 native surface samples had finite skin positions, positive triangle
areas, and zero skin vertices more than 1 mm below the bed. That lightweight online
surface audit does not detect surface intersections. Exact intersection results in this
bundle pertain to the four-second poses. Continuous clearance, full deformable skin/fat
contact, and the pleural-pressure-enabled scene remain unqualified.

`motion-metrics.json`, the compact coupled trace, and the exact witness tables bind these
results to their launch and capture hashes. `lossless-capture-recovery.json` names the
on-host compressed geometry and SHA-verified restoration commands. The trials did not
alter the published default scene.
