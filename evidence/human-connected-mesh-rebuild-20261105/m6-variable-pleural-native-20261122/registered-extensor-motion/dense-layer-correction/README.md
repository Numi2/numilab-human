# Bilateral extensor repair with measured skin and fat boundaries

The initial bilateral fit improved compact motion and made two interfaces worse on the dense left forearm. This revision keeps the tested right binding and refits the left donor rotations about the existing named force sites. The objective includes both compact poses and the accepted dense outer-skin, inner-skin and deep-fat boundaries. The separate eight-second trajectory remains a held-out pose. All comparisons use actual emitted native geometry. No previously clear anatomical bank acquired a crossing in these captures.

| Same-pose measurement | Unchanged control | Revised fit |
|---|---:|---:|
| Both extensors against compact skin at entrance | 840 | 642 |
| Both extensors against compact skin at 8 s | 1,275 | 1,063 |
| Both extensors against compact skin at 16 s | 1,235 | 1,084 |
| Both extensors against dense outer skin at 31.25 µs | 2,386 | 1,490 |
| Both extensors against dense inner skin at 31.25 µs | 2,904 | 2,771 |
| Both extensors against dense deep fat at 31.25 µs | 2,374 | 2,149 |
| Whole-body compact skin/anatomy at entrance | 8,859 | 8,661 |
| Whole-body compact skin/anatomy at 16 s | 14,064 | 13,913 |
| Whole-body dense skin/anatomy at 31.25 µs | 11,791 | 10,895 |

The left dense outer/inner/deep counts are now 575/1,194/993 against 592/1,271/1,187 in the control. The right counts are 915/1,577/1,156 against 1,794/1,633/1,187. At compact 16 s, the two extensors' self-intersecting triangle pairs fell from 881 to 704, their bone crossings from 1,660 to 1,352 and their crossings with neighboring muscles from 2,782 to 1,995. Some already intersecting structures worsened. Right extensor carpi radialis brevis gained 102 pairs at 8 s and 55 at 16 s. Left extensor indicis gained 12 pairs at each of those frames. The full per-bank witnesses remain accessible.

Each side retained its source material coordinates, connectivity, sparse weights, 19 donor identities and 36 named route sites. Maximum measured named-site displacement after binary32 emission was 0.100 µm on the right and 0.071 µm on the left. All 877 other geometry banks retained their coordinates and topology in the matched captures. All 77 columns of the 250-sample compact physiology trace and the separate contact-impulse trace matched byte for byte. These changes register passive envelopes. They do not establish volumetric muscle force fidelity.

## Integrated motion and material validity

SSH Numi, Mac mini, Apple M6, 16 GB unified memory. The compact native model accepted 8,000 steps of 2 ms with Brain control, mechanical respiration, circulation and variable pleural pressure. Over 16 simulated seconds it completed three breaths and 19 cardiac filling/ejection cycles. Pleural pressure ranged from −770.01 to −470.34 Pa. Body runtime was 156.4755 s (0.10225×). Cold launch through native return took 165.3298 s (0.09678×). There were 250 accepted presentation callbacks. The compact skin had 17 self-intersections at entrance and 40 at 16 s. These are sampled numerical checks.

The dense native model accepted two steps totaling 31.25 µs in 203.618 s cold (0.0000001535×). At that endpoint the 705,090 skin cells and 743,709 fat cells passed the installed orientation and quality thresholds. The anatomy is a mixed-source reference adult. Sustained dense motion and supported rest remain unverified. The original discrete geometry and native acceptance reports are in the recovery archive.

The paired comparison can now carry forward the unchanged part of an existing whole-body clearance audit. It authenticates the control frame, confirms every other bank's geometry is byte-identical, and requires the original counts for both muscles to agree with an independent audit. Then it updates the two affected counts. Separate full scans confirmed the results for the compact entrance, compact 16 s and dense endpoint. Matched comparisons took 2.71–2.89 s in these three cases. Full independent scans took 40.35, 41.54 and 100.50 s respectively. These timings describe geometry verification.

## Why the dense body has not settled

The existing preparation predicted 953.18 N from the initial flat-bed barrier against 952.86 N body weight. At the two-step endpoint the native bed reaction was 3,491.91 N. Re-evaluating the same elastic barrier on the accepted physical nodes gives 3,485.30 N, within 0.19% of the native root-averaged reaction.

The torso contribution rose from 176.42 N across seven active nodes to 2,686.23 N across four. Those four nodes account for 77.1% of the terminal elastic reaction. Pelvic loading remained close to 764 N. The largest torso witness moved 1.04 µm toward the plane within its 10 µm barrier band and its estimated load rose from 94.73 N to 1,038.60 N. On the matched reference, the previous native force observer measured only 17.87 N of vertical FEM loading at the articulated root against an 811.45 N root gravity target. The current dense trial reproduced the reference's exact physical skin/fat coordinates, body state and contact trace. The archived reaction calculation documents the distinction between static elastic and native root-averaged forces.

Keep the corrected registration as a regional improvement. The physical release criterion is a supported equilibrium that transfers torso and pelvic loads through tissue to skeleton and remains stable through motion. The initial total bed force alone gives a misleading impression of balance. The exact witnesses above allow a subsequent coupled equilibrium solve to be evaluated locally and globally.

## Replay

Start from the parent candidate's verified recovery files and its named route registry. Run the following on Numi with the same native dependency paths as the parent. The archive includes the tested programs, launches, edited binary payload, actual endpoint packs, traces, failed scaled-donor attempt, installed material positions and independent validations. Its recovery receipt records the digest of each independently downloaded member.

```sh
python3 fit_extensor_material_clearance.py "$entrance" "$terminal_control" "$dense_control" "$reader" "$predicate" "$routes" "$initial_fit" "$physical_fit"
python3 install_extensor_bindings.py "$drive_control" "$physical_fit" "$routes" "$physical_registered"
NUMI_RETAIN_SUPPORT_GEOMETRY=1 NUMI_CAPTURE_BED_COLUMNS=1 python3 run_bed_work.py "$physical_registered" "$footprint_owner" "$physical_16" 8000 --disabled
python3 verify_native_extensor_pair.py "$terminal_control" "$physical_16" "$reader" "$predicate"
python3 verify_extensor_material_layers.py "$dense_control" "$physical_dense" "$reader" "$predicate"
```
