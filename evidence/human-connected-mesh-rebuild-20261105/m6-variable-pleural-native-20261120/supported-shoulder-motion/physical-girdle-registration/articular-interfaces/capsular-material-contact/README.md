# Bilateral shoulder material registration on Apple M6

The named superior glenohumeral entheses now map to four explicitly identified triangles in the current NHBONES1. The precise native bone arithmetic matters here. The binary32 scale product and the quaternion stored in the asset are evaluated using the same ordering as the C++ baker. The largest discrepancy over the four reconstructed body-local attachment locations was 3.47e-17 m.

The Human authoring entry point is `python -m numilab_human.capsular_material_registration`. It verifies the atlas archive, member hashes, owner and laterality, emitted face ancestry and the calibration reference. It minimizes the distance to each registered MyoSim site in body-local space within the original atlas attachment footprint. The result uses the existing material-face and barycentric fields consumed by the native ligament catalog. The scapular and humeral proxy regions were inferred on a mixed-source adult. The reference is not a measured individual. A human cadaver study describes the superior labral and lesser tubercle attachments. [Kask et al.](https://pubmed.ncbi.nlm.nih.gov/20434927/)

| Side | Scapular face | Humeral face | Donor registration residual at entheses |
| --- | --- | --- | --- |
| Left | 11 / 7360 | 13 / 526 | 0.794 / 0.214 mm |
| Right | 10 / 19072 | 12 / 2951 | 0.669 / 0.301 mm |

## Physical admission

The full executable built and ran on SSH Numi. It stopped in the straight-ligament constructor with zero accepted physical steps. The three independent entrance cooks admitted 32, 33 and 32 of 34 rows. Bilateral ligament paths are still required. The physical skin/fat, clavicle and tendon assets used the preceding scene revision.

The elevation and rotation axes share a pivot in the donor model. Across 255 configurations the two articulated bodies reconstruct that common point to within 0.00008 mm. Independently transferring it through the two atlas registrations separates the inferred centers by 42.51–45.09 mm on the left and 43.25–45.43 mm on the right. The reverse triangle inequality gives rotation-only lower bounds of 38.69 and 38.80 mm about a common pivot, apart from the measured submicrometre pivot discrepancy. Pose adjustment alone cannot align these two registrations. The 255 configurations comprise default, three catalog entrances, the full-scene captured entrance and 250 reduced-control snapshots.

The mechanical reference places the two named scapular sites 44.70 and 44.63 mm away from their mapped counterparts in the registered body frames. The corresponding humeral differences are 5.92 and 4.53 mm. Replacing the atlas bones with the donor visual meshes would introduce open and nonmanifold edges on both paired bones. The donor meshes themselves have zero joint intersection pairs in eight configurations. Projecting the named ligament sites directly onto that donor geometry still leaves a path crossing at each configuration.

At the full scene's bed entrance, two exterior paths supported by actual bone features passed independent geometric and unit-tension virtual-work checks. Their lengths are 46.52 and 45.91 mm. The verifier evaluates open spans against both owning solids and all other bone banks, checks nonnegative normal contact reactions, and tests six independent rigid perturbations. Actual material ancestry identifies boundary spans shared with a bone face. These are binary64 authoring calculations. They establish neither physiological resting tension nor a native dynamic force history.

At recorded control step 256 the continued left path measures 50.112 mm and fails clearance and local contact equilibrium. A reselected path passes the static force test at 39.555 mm. Installing this discontinuous choice in the running solver would change stored elastic energy without a demonstrated physical transition. The compiler therefore leaves native force ownership unchanged. The next structural repair must reconcile the glenoid reference with the articulated shoulder, skin, fat, muscle attachments and bed support together.

The direct-search summary retains the bounded 312,912 sampled pair outcomes. The shared-frame evaluation includes all eight sentinel poses. At the midpoint, a right shoulder collision surfaced in an entrance outside the original five-pose comparison. These figures are geometric evidence on recorded poses. Whole-body crossings, sufficient settling, variable pleural regulation and full-scene throughput remain unqualified.

## Reproduction

Run `./reproduce.sh` on SSH Numi from the Human checkout. It uses the retained sources and the previous articular repair. It regenerates the four sites, recompiles the checked native catalog, independently compares C++ material positions, re-solves the stored material branches and launches the real complete executable. The expected native result is a rejection before the first step. The compact receipts preserve branch failures as well as successes. No patient geometry or new runtime dependency is embedded here.

BodyParts3D © The Database Center for Life Science, licensed under CC Attribution-Share Alike 2.1 Japan. The preceding source registration records the donor member and archive identities.
