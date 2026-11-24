# Supported human on Numi

The detailed physical model starts in a dynamic release. Three outer skin nodes lie within the bed contact range. The initial static bed reaction is zero and the reported maximum acceleration imbalance is 68.31 m/s². The native bed and tissue contacts are enabled. This is an unresolved initial condition for a supported adult.

| Matched native measurement | Result |
| --- | --- |
| Detailed material import | 314,013 skin/fat nodes and 1,457,313 tetrahedra. Zero initial inverted or low-quality cells. Source mass replay passed. |
| First detailed physical step | 7.8125 µs accepted. No inverted cells or crossings among the four material surfaces. Four skin and 26 fat cells fell below the shape gate. The 185-bone solid audit found 5.977 cm³ of summed tissue/bone overlap. |
| Second-step equilibrium | The full-body control rejected at 0.00152729. Independent GPU row reduction placed 99.99998% of the squared residual in skin/fat mechanics and 27.67% at one shared node near the left ulna. |
| Represented direction repair | The guarded full-body replay accepted the full sixth Newton correction after the revised Metal solve. All 32 fractions had previously failed the geometry test. The final equilibrium certificate still rolled back the step. |
| Reduced contact experiment | 16 seconds in 178.67 wall seconds (0.08955×), 3 breaths, 19 full filling/ejection cycles. The last second averaged 952.87 N of bed reaction with zero root assistance. The endpoint had 11,623 skin/anatomy triangle intersections. The control took 172.37 wall seconds and had 11,543. |

**Implementation decision.** Retain the demonstrated Metal geometry repair as an opt-in increment. Keep failed material fits and the pelvis contact experiment out of the default scene. The next physical target is to initialize the *detailed* adult in joint balance under gravity, skin/fat prestress, skeletal attachments and unilateral bed contact. An admissible cell and a low stored energy do not establish equilibrium. Force balance and registered material validity must hold in the same native transaction. The fitting experiments and the material-volume safeguard must never supply a fictitious physical reaction.

The reduced path remains a breathing and circulation comparison. The detailed path remains the anatomical and mechanical reference. Neither qualifies as the delivery scene. The 16-second zero-crossing target at 0.1× and the longer baseline/intervention qualification remain open.

[Physical skin and fat](../m6-conservative-interior-motion-20261124/README.md) · [Native equilibrium attribution](../m6-native-equilibrium-attribution-20261124/README.md) · [Connected pose fitting](../m6-registered-pose-equilibrium-20261124/README.md) · [Accepted motion and cortical overlaps](../m6-load-margin-check-20261124/README.md) · [Represented Metal direction](../m6-represented-volume-direction-20261124/README.md) · [Reduced support trial](../m6-pelvic-bed-sections-20261124/)
