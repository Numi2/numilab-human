# Distributed forearm support in the coupled resting human

The condensed bed model integrated skin and fat column stiffness over each anatomical section, then applied the section's reaction at one selected skin point. A source-binding trial preserved every section stiffness but changed the selected left forearm vertex from 49166 to 49152 at step 352. The sampled support impulses first differed at step 544. Protecting the sampled footprint alone increased terminal skin/anatomy crossings from 13,529 to 14,153.

The native implementation partitions each registered forearm footprint into four disjoint subregions by its projected tributary area. Each subregion has its own existing GPU contact point, Jacobian and unilateral response. The two named NHCNT source rows and both anatomical parents remain identifiable. All 7,282 input columns retain the same projected area, skin and fat thickness, and assigned material parent. Subdivision preserves the original total force under uniform compression to a relative error below 2e-8. The six new contact rows carry no additional mass or initial load.

| 16-second result on M6 | One point per forearm section | Four forearm patches |
| --- | ---: | ---: |
| Right maximum sampled change in loaded center of pressure | 49.71 mm | 26.49 mm |
| Left maximum sampled change in loaded center of pressure | 72.84 mm | 21.95 mm |
| External pairs at 7.998 s | 11,438 | 11,502 |
| Skin self-intersections at 7.998 s | 33 | 32 |
| External pairs at 16 s | 11,615 | 11,543 |
| Skin self-intersections at 16 s | 24 | 24 |

A maximum of two independent patches on each forearm carried load simultaneously. These pressure-center comparisons use consecutive loaded samples 64 simulated milliseconds apart. The largest movements decreased while the median movement increased slightly. The final second's sampled vertical bed reaction averaged 952.71 N against 952.86 N of body weight. Peak COM speed was 0.090 mm/s in that interval. Across the two measured poses one skeletal body still moved 8.69 mm. Three breaths and 19 complete filling/ejection cycles occurred under variable pleural pressure.

The four-patch run used the connected elbow skin trial. Compared with the same skin on 32 sections, 23 anatomy banks gained crossings at 16 s. Compared with the previous default (13,529 physical pairs), the new run had 1,986 fewer pairs. Remaining surface defects include partially external extensor origins. The 15.936 s active interval achieved 0.10012×, and cold native execution took 172.37 s (0.09282×). Whole-body contact and viewer frame cadence remain unqualified.

The volumetric reference still rejects its fourth step. A local solve added an interior point to each candidate homogeneous fat pocket and equilibrated it under the existing constitutive energy. Across both witnesses 63 candidate configurations passed the geometry checks at release and accepted displacement. The best boundary force differences were 34.7% and 79.6% under those two loads. The independent affine check recovered the original boundary tractions to roundoff. No volumetric remesh was installed.

The recovery receipts identify exact input and executable hashes, independent downloads and member checks. The physical owner can be rebuilt by applying the included installer to the pinned compact native owner. The original 32-section model remains available as a comparison.

The ordinary 38-section trajectory reproduced all captured physical states and six traces with either cached contact Jacobians or the first-SIMD sweep. Caching gave 0.09212× cold. The sweep gave 0.09339× cold compared with the original 0.09282×. The small timing separation across single runs does not establish a speedup. The M6 launcher `~/Work/run-numi-human-rest --forearm-contact` selects the measured four-patch configuration. Its existing no-argument control remains available.

At release, the two failing dense fat cells have opposite volume acceleration contributions from the surrounding dermis. Under routed lumped mass the elastic contributions are −720 million and +1,874 million s⁻² respectively. About 99% of the absolute component magnitude comes from skin in both cases. This accounts only for constitutive forces in those two complete bonded stars. The full native step also solves damping, contact and other constraints.
