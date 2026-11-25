# Posterior lumbar anatomy and support on Apple M6

Run the dense scene on SSH Numi with the bilateral lumbar surfaces and the registered shoulder tendons.

```sh
bash scripts/run_resting_human_on_numi.sh --dense --posterior-support registered-lumbar --steps 2
```

The launcher verifies 501 pinned files. The current native run accepted two 7.8125 µs steps with variable pleural pressure. Elapsed time including initialization and export was 218.24 s. The run exercised 15.625 µs of physical time. Supported motion over 16 s remains unqualified.

The two iliocostalis lumborum surfaces use original BodyParts3D members FJ1527 and FJ1527M, registered through 18 matching native vertebral and sacral surfaces. At the accepted endpoint both muscles had zero self-intersections and zero crossings with nearby anatomical banks or the skin/fat volume. The left airway surface has open ends. A separating plane proves at least 6.67 mm clearance from its complete convex hull. The active muscle, registered skin/fat, cardiac and controller owners were retained. Bed impulses, physical skin/fat positions, muscle force observations and coupled physiology were byte-identical to the control. These two surfaces currently add no force or mass. [Original anatomical data](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/data-7.html)

The skeleton accelerated toward the bed at 9.798 m/s². The 2,216 active skin/fat-to-bone contacts supplied 1.438 N along the support normal. At their observed geometry and material slopes, even freely redistributing compressive pressure would require at least 585 mm peak closure to balance the skeletal load with the other owners held fixed. A separate cut-interface calculation found that assigning the previous 823 N lumbar preload to the two new muscles through cortical compression alone would require at least 25 MPa of peak pressure. These bounds explain why increasing the preload in this region gives a misleading starting state.

The immediate mechanical work is to recruit support across the posterior body and transfer it through registered adipose, passive muscle and its skeletal attachments. Four of the 24 native iliocostalis paths enter their respective registered muscle at this pose. Complete force and torque closure, admissible material strain and native bed reaction are necessary checks before treating any proposed initial coordinates as supported rest.

The terminal material audit still fails. Its witnesses and all endpoint measurements are retained in [the native result](./registered-lumbar-motion.json). The original anatomy, failed Float32 emission, repaired geometry and exact native executions have independently verified recovery records in [the authoring archive manifest](./registered-lumbar-authoring-recovery.json) and [the native archive manifest](./registered-lumbar-native-recovery.json).

## Physical registration of the lumbar interfaces

The registered muscle surfaces meet the physical fat volume on two right-hand triangles and eleven left-hand triangles. Refining the full adjacent tetrahedral stars and the interior of the loaded faces increased the receiving fat nodes from 4 to 124 on the right and from 13 to 45 on the left. The median of the longest triangle edge over right-hand quadrature samples fell from 151 mm to 40 mm. A 151 mm edge remains at the edge of the right patch. The local transfer reproduces both stored loaded states with less than 1.1e-12 N force difference and less than 2.6e-14 relative strain energy difference.

The native reference-volume importer accepted all 1,457,778 tetrahedra. It registered 3.452571 kg of skin and 10.487108 kg of fat, with zero inverted cells, low-quality cells, density outliers or unassigned nodes. Original donor tetrahedra resolved 7,027 inherited face-label discrepancies. Every final cell has a winning P1 tissue group and a supported source-face identity. The native source-carrier allocation passed separately for both tissues. The physical bed-load test still uses the preceding volume topology.

The two prospective passive muscle volumes were tested at 8, 4 and 2 mm coordinate spacing. The 4 to 2 mm change in strain energy exceeded the 5% engineering gate on both sides. The registered muscles remain passive anatomy in the coupled run. [Physical registration and convergence results](lumbar-material-transfer.json)
