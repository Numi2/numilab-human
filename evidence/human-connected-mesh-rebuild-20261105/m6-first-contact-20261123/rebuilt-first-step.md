# First coupled GPU step with the rebuilt physical skin and fat

On SSH Numi the complete native human admitted the registered 311,361-node skin/fat volume, replayed both source mass allocations, applied gravity and advanced its mechanics and physiological clock by 15.625 µs. The release had the same bed height, the same initial gap of 1.948 µm at the same outer node, all 14 original deep attachment positions, and zero added initial forces. The initial installed geometry passed the native inversion and shape checks in all 1,450,187 material cells. The independent endpoint anatomy and material checks failed after the step. [Endpoint](rebuilt-first-step-endpoint.json) · [Physical measurements](rebuilt-first-step-mechanics.json)

| Endpoint measurement | Result |
|---|---:|
| Bed reaction / gravitational weight | 0.47884 N / 952.86443 N |
| Peak physical skin/fat nodal displacement | 45.68 µm |
| Skin-to-muscle, skin-to-vessel and skin-to-tendon triangle pairs | 9,264 / 792 / 76 |
| Skin-to-bone pairs, skin self-intersections, skin/fat layer crossings | 0 / 0 / 0 |
| Inverted skin/fat cells | 0 |
| Fat cells below the shape minimum | 2 |
| Fat cells outside the existing principal-stretch range | 12 |
| Distinct material cells failing either motion check | 13, all in the rebuilt forearm fat |
| Time from launch to exit / native body timing | 157.34 s / 15.69 s |
| Corresponding real-time factors | 9.93e−8× / 9.96e−7× |

The most distorted cell had a reference edge-matrix singular value of 0.101 µm and an endpoint maximum principal stretch of 29.53. Another cell fell to 35.27% of its reference volume. A positive determinant at release does not qualify these cells for motion. The exact failing cell IDs, node IDs, deformations and displacements are preserved. All registered skin triangles were checked against 878 anatomy banks at the endpoint. Counts denote intersecting triangle pairs. [Cell witnesses](rebuilt-first-step-fat-witnesses.json) · [Crossing classification](rebuilt-first-step-crossings.json)

The native volume-direction solver had identified two tissue cells by their former positions in the element array. Both ordered tetrahedra, their materials and their reference coordinates survived the rebuild. The updated runtime checks both witnesses at the emitting object's local rows, verifies their common owner, binds the current topology hash and retains the existing release bounds and numerical solver. The earlier unmatched runtime correctly rejected initialization. The original material control reproduced exactly in the updated measurement tool.

The compact supported scene previously completed 16 seconds at 0.09267× from launch to exit with variable pleural pressure. It still had 11,527 physical skin-to-tissue crossing pairs at the endpoint. Its heartbeat and breathing evidence belongs to that model. The full scene above started from first contact, with bed reaction at 0.0503% of body weight on the measured step. Neither result qualifies the requested sustained, settled whole human. [Compact and dense shoulder measurements](shoulder-physical-motion.md)

For the resting-body qualification, establish a reproducible supported equilibrium with measured bed loads reaching the skeleton, repair the thirteen motion-sensitive fat cells and clear the registered skin-to-muscle, vessel and tendon penetrations. Use that equilibrium for matched full and reduced motion runs before comparing sustained speed. [Physical mass recovery chain](current-role-mass-replay.md) · [Independently restored native binary, accepted state, launch inputs and audit witnesses](rebuilt-first-step-recovery.json)
