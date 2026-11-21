# Load and motion baseline at the full reference mass

The existing native reduced scene on SSH Numi accepted 8,000 consecutive 2 ms steps with live variable pleural pressure. The tested model changes only the declared 72 kg reference mass to the same 97.13195069110952 kg source-composite mass used by the latest fine-volume scene. This removes the uniform mass and inertia scaling and leaves the contact law, layer sections, anatomy payloads, Brain, physiology, timestep and acceptance settings pinned. The run starts in the released pose. The initial geometry cache was restored byte-for-byte from the retained compressed pack. Startup decoding occurs before the process timer.

| Metric | 72 kg retained control | 97.13 kg replay |
| --- | ---: | ---: |
| Simulated seconds accepted | 16.00000076 | 16.00000076 |
| Late sampled bed reaction | 706.366 N | 952.907 N |
| Late sampled bed impulse divided by gravity impulse | 1.000065 | 1.000045 |
| Terminal centre-of-mass speed | 0.29842 mm/s | 0.01433 mm/s |
| Viewer-inclusive motion throughput | 0.10739× | 0.10520× |
| Process launch-to-exit throughput | 0.10024× | 0.09800× |

Both runs report zero root force and torque assistance and zero mass residual. The 97.13 kg run completed three breaths and nineteen modeled filling/ejection cycles. Measured pleural pressure ranged from −770.014 to −470.340 Pa. All 251 surface samples pass the existing finite coordinate, face area and bed penetration checks. The maximum sampled functional cardiac volume discrepancy is 2.3374e-5 relative. The measurements use the same individual visible-scene preparation and accepted-state diagnostics as the retained control and qualify those diagnostics only.

## Development decision

Use this mass-matched scene as the integration control for released settling, support loads and coupled physiology. Keep the dense mesh and its conservative mass map for geometry and material reference tests. The fine-volume native prefix accepted 46.875 µs in 179.24 seconds with zero accepted bed load. More dense-solver tuning from that release cannot establish a resting operating point. The coarser 28,632-cell experiment also carried less than one newton during its 2 ms release check. A lower element count alone did not fix the load path.

The current control represents skin/fat compression through 32 physical support sections. Whole-body skin/fat stress, tissue-to-tissue contact, the 26 FEM fascia regions, and the 34 registered ligaments are outside this run. Incremental tissue additions should keep the same physical mass, use the existing force and accepted-state owners, and demonstrate regional force and deformation agreement against loaded local reference tests. Connect the skin, fat, tissue and skeleton interfaces during authoring. Test their combined motion with exact intersection predicates before extending the benchmark. The accompanying hand fit demonstrates why checking positivity and shape alone is insufficient for this decision.

The model still has unresolved interior crossings. That did not change during this test and the new mass variant had no full internal clearance export. The qualification target remains 16 seconds at or above 0.1× from launch to exit with zero erroneous crossings, coupled tissue and ligament motion and varied pleural pressure. Longer physiology and intervention tests remain open.

## Reproduction and evidence

Place the two replay scripts next to the original supported-control and fine-volume checkpoints under the same Work tree. Run /bin/bash run.sh on SSH Numi. The launcher verifies the exact source-mass and asset inputs and uses an automatically ejected RAM-backed scratch filesystem. The versioned launch JSON records the replay configuration and source identities. The compact physiology, bed contact and surface traces belong to the same process. The full original logs and complete support impulse stream remain on SSH Numi. The compare.py check reports the dimensionless sampled bed impulse ratio after normalizing the segment gravity impulse to one physical step.
