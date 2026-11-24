# Right shoulder material and registration decision

The complete native frame in the preceding experiment placed 67% of its terminal bed impulse on one overlying skin node. The reference muscle there is the right infraspinatus, stable ID 83, BodyParts3D member FJ1500 and MyoSim actuator INFSP. Its existing source donors are the right humerus and scapula. This package preserves surface identity, bindings and the active muscle owner.

## Material construction

TetGen ran on the exact welded source surface on SSH Numi. One interior tetrahedron inverted under the existing donor registration. Replacing three tetrahedra with two restores positive orientation across all 1,448 cells. All 866 original boundary triangles and all 479 material vertices are preserved. The lowest determinant ratio along the straight interpolation from the opening state to the two-step accepted state was bounded below by 0.179057. This covers 15.625 µs. Surface volume agrees with the captured muscle to better than 0.0001 cm³. Determinant positivity alone does not qualify the tissue law or attachment.

## Contact and surrounding tissues

All 185 cortical banks were covered by the distance and solid tests, with spatial exclusion for separated banks. At the accepted endpoint the muscle penetrates the scapula by 0.2514 cm³ and the humerus by 0.6938 cm³.

A direct partition preserves the source-face ancestry of the resulting cortical interfaces. It splits off 1.188 mm³ and displaces 945.264 mm³ of muscle. The partition is retained for attribution.

A connected fit conserved total muscle volume and reduced both bone intersection volumes below 3e-9 mm³. Its maximum material motion was 4.775 mm. Checking 74 neighboring displayed solids exposed a new 1.39 mm³ acromial deltoid intersection and increased spinal deltoid and teres major intersection volumes. This disqualifies the bone-only fit.

A second fit considered surrounding fixed tissues together and retained positive relative material determinants. Its largest point correction was 18.73 mm. It still left 653.3 mm³ of ascending trapezius overlap and increased total volume by 2666.5 mm³. This candidate also failed admission. Both fits are authoring experiments.

## Registration ownership

The eight-bank comparison preserves the stored source coordinates, neutral skeletal reconstruction, supported opening reconstruction and actual accepted native triangles as separate stages. Pairwise volumes may double-count occupied space. They are not tissue mass estimates.

| Infraspinatus partner | Stored mesh coordinates | Neutral skeletal reconstruction | Accepted native geometry |
| --- | ---: | ---: | ---: |
| Right ascending trapezius | 0.000 cm³ | 10.714 cm³ | 10.794 cm³ |
| Right teres major | 4.777 cm³ | 5.077 cm³ | 6.212 cm³ |

The trapezius mismatch is already substantial in the skeletal reconstruction. The teres major overlap also exists in the stored mesh inputs. The results include separate current body and muscle owners, rotations and reconstruction errors. The two trapezius surfaces differ from the simple donor reconstruction at the accepted state by up to 41 µm. The accepted column uses the actual native surfaces.

Further shoulder qualification requires a common material registration for these interacting structures with the humeral and scapular attachments preserved. Carry the right infraspinatus volume as a reference asset. Keep the current simulator as the control until the combined material, geometry, mass transfer and load response pass. None of the shoulder candidates in this archive altered the native run.
