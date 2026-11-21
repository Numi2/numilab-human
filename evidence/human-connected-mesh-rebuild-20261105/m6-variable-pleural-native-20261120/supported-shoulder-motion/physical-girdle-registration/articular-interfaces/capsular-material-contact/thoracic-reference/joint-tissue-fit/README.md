# Joint fit of skin, fat and skeletal clearance

The candidate keeps the original material connectivity, node identities and source-carrier ledger. One sparse correction moves the actual volume nodes. Its rows include opposing bone/material triangles, within-layer and between-layer triangle pairs, signed cell quality at Float32 emission, and receiver density capacity. Tissue/tissue rows act on both sets of material corners. The bone rows keep the rigid bone surface fixed. Every reported result comes from the same restored TetGen node file.

| Entrance measurement | Prior connected-volume reference | Joint fit |
| --- | ---: | ---: |
| Outer skin / inner skin pairs against 62 moved bone banks | 0 / 0 | 0 / 0 |
| Deep fat pairs against those banks | 1,347 | 580 |
| Self-intersections of outer skin / inner skin / deep fat | 236 / 0 / 690 | 4 / 0 / 373 |
| Crossings between separate material layers | 6 | 0 |
| Inverted or below-limit cells across both tissues | 0 | 0 |
| Receiver groups outside density bounds | 53 | 50 |
| Summed local density-capacity violation | 7.252 g | 7.012 g |

The 580 deep-fat pairs remain entirely at the two scapulae and humeri. The other 58 moved bone banks clear all three tested tissue surfaces in this entrance pose. The total material mass is 13.93967896346 kg. Its 71,639 source quotas are individually conserved within 1.74 × 10⁻¹⁸ kg. Receiver failure and unverified spatial/P1 ancestry still preclude installing this mass mapping. The positive cells also include large local changes in reference volume, reported per material in `material-transfer.json`. They qualify neither a stress-free anatomical reference nor deformation during motion.

The solver first selected local source-rigid corrections only when they introduced no new crossing or invalid cell. It then processed the remaining skeletal, tissue and density witnesses together. The fit rechecks actual native Float32 volumes, three physical boundaries against all 62 moved bone banks, all six pairs of tissue boundaries, and the native mass solver. Independent integer-lattice checks reproduce all 377 remaining erroneous tissue pairs.

The recovered coordinate diff applies to the preceding standard TetGen node file. Run `./reproduce.sh` on SSH Numi with the existing retained inputs. It verifies the candidate identity, exact native volumes and contact results; the production scene continues to use its prior assets. The 16-second full-scene trial with variable pleural pressure and its end-to-end performance target remain open.
