# Supported shoulder motion on SSH Numi

The right infraspinatus now applies force through four cortical sites on the right scapula. The original site was 30.90 mm from the bone. The final re-registration moved it 52.20 mm and changed the same-body terminal segment length by 0.30 nm in the regional model. Source triangle and vertex identities, connected geodesic selection, the native tendon force map, tendon architecture and body mass are retained. This is a mixed-source anatomical reference.

The first moving footprint lost coverage in eight early sampled poses. The final compiler search followed the actual source-length sphere, identified 29 eligible centers from the selected moving poses, and searched the connected infraspinatus-facing cortical faces. The four selected sites and tendon endpoint passed at all 250 saved body-motion samples and at both authoritative native geometry captures. The right infraspinatus had zero nonadjacent self-intersections in those checks. The intermediate body poses use the native program’s CPU kinematic evidence output. This sampling leaves inter-sample and whole-body clearance open.

| Native measurement | Result |
| --- | ---: |
| Accepted steps at 2 ms | 8,000 |
| Simulated duration | 16.0000 s |
| Simulation wall time | 163.368 s |
| Cold process wall time | 172.343 s |
| Simulation / cold real-time factor | 0.09794× / 0.09284× |
| GPU duration summed across 250 simulation segments | 149.222 s |
| Reported presentations, including startup and final | 252 |
| Normal bed reaction averaged over 250 sampled physical steps | 952.566 N |
| Represented body mass | 97.13195 kg |
| Root assistance | 0 N, 0 N m |
| Complete breaths recorded / sampled LV pressure peaks above 70 mmHg | 3 / 19 |
| Final tidal volume / cumulative aortic ejection | 528.82 mL / 1,352.35 mL |

The native physiological and body clocks matched with variable pleural pressure. The eleven compared respiratory and cardiovascular channels retained their control values. The support audit exports the impulse of the last 2 ms step in each 32-step segment. Its 250 samples cover the 16-second interval. The surface audit reported zero nonfinite skin vertices, zero vertices more than 1 mm below the bed, and maximum functional volume relative error of 2.34e-5. The contact trace reported up to 6.92 mm of penetration at the modeled compliant sections.

The executable uses the established normal skin/fat bed-section law with area, layer thickness, moduli and damping. Section mass is included in articulated donor inertia. Passive whole-body fascia is disabled in this configuration. Average support near body weight, a valid shoulder footprint, and repeated pumping establish separate measured properties. They leave dense whole-body settling, all internal crossings, clinical accuracy, 15 wall-clock frames per second, and long-duration gas regulation open.

## Whole-body skin through motion

The external skin fails the anatomical release checks at both native captures. The same established Float32 audit evaluated all 878 target meshes. None of the observed pairs was classified as ocular.

| Accepted time | Skin-to-structure triangle pairs | Skin self-intersection pairs | Target meshes touched |
| ---: | ---: | ---: | ---: |
| 7.998 s | 11,514 | 32 | 86 |
| 16.000 s | 11,531 | 24 | 86 |

At the final capture, 8,869 pairs involved muscle surfaces and 2,217 involved bones. The remaining 445 belonged to semantic class 51011. The two captures shared 8,211 exact face pairs, while 3,303 disappeared and 3,320 new pairs appeared. These are triangle-pair counts. Distinct affected skin faces numbered 2,490 and 2,543. Both complete native geometry packs from the independently executed launcher repeat were byte-identical to those audited. The [recovery record](skin-audit-recovery.json) identifies the complete face-level evidence.

Use this configuration as a repeatable benchmark for bed loading, physiological coupling and the registered shoulder attachment. Whole-body anatomical release remains blocked by these observed defects.

Both skeleton registrations contain the same scapular cortical triangles. Their per-vertex residual after the fitted rigid transform is below 3e-16 m. The dense and regional payloads use the same four cortical vertex identities. The dense native scene with this force attachment accepted two 7.8125 µs steps, then rejected step three. Its retained fresh nonlinear residual was 0.0317013. At the last Newton iterate the volume lower bound on global cell 140978 restricted the step fraction to 1.19209e-7. Fresh mechanical, contact and material acceptance remain in force. An attempted transplant of the latest dense muscle surface into the regional moving poses produced 53 self-intersection pairs at each of the two captures. That tissue transplant was excluded from the running model. The dense volume still has four skin and 33 fat shape failures in the measured reference prefix.

Run the verified regional scene on Numi from its installed directory. The launcher verifies 80 pinned inputs, writes each run to a fresh directory, and uses the existing native executable. Launch configuration, source geometry, physics owners and timestamped output are retained beside the executable inputs.

~~~sh
ssh numi '~/Work/numi-human-supported-shoulder-085/force-route-086/run-supported-scene'
~~~

The launcher repeat accepted all 8,000 steps, and the 250-sample and two-capture footprint checks passed again. Cold process wall time on repeat was 173.80 s. The independently restored recovery archive contains both native traces, input registrations, load mappings, exact replay geometry and the dense rejection. Its parent archive retains the native executable and the failed moving-surface trials.
