# Left forearm diagonal repair verified in the native scene

One same-parent diagonal change removes the three known late-pose forearm boundary crossings in the ordinary-path CPU reproduction. Source vertices, exact surface coverage and signed volume, the shared interface, and deformation coefficients are unchanged. Only left muscle 132 faces 5111 and 5112 change.

Native012 ran the complete existing resting scene on SSH numi (Apple M6). All eight captured states pass both left child self-intersection checks and their joined-boundary check, with zero degenerate faces. All 4,588 shared vertices are bit-identical; 7,010 opposite interface facets cancel. The joined boundary retains 12 closed components and its winding signs. Forearm vertex coordinates are bit-identical to native009; all 866 other surfaces, 157 accepted body poses, and 1,000 physiology trace rows also match.

The run accepted 8,000 steps / 16.00000076 simulated seconds in 106.59056649999911 seconds of launcher wall time, approximately 0.15 times real time. The continuous recording is in viewer/native-viewer.mov. Its end frame shows the whole supine adult on the existing flat support, with 3 breaths and 19 completed cardiac filling/ejection cycles. Recorded root assistance force and torque are zero. These observations do not establish complete physical or physiological qualification.

Launch the exact retained scene on numi:

    /Users/n/numi-human-prep-venv-20261005/bin/python3.13 /Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/forearm-shared-field-native-012-diagonal097/launch_arm.py baseline

The 89 immutable launch inputs were checked before and after execution. The declaration identifies the source revisions and binary. The unchanged native executable SHA256 is 147c413bdfa84687f4afa3ce4e0d37576a853a3b4d5f421866e4faefeab4f174. Executed scripts, exact source-change proof, native checks, compact traces, and external asset pins are retained here.

The changed triangles each cover portions of both old row-local faces 5819 and 5820. The exact pooled-lineage overlay in source097 corrects the retained scalar intermediate-face ancestry. The original anatomical source parent remains unchanged; no source identity is discarded.

The older step151999 result is a complete CPU geometry check using a retained pose and ordinary shader arithmetic. It is not a fresh five-minute native run or an all-pose admission. Source099 and source-097-change-proof supersede the unstable local-plane Float32 metric in source097: the exact rational patch is coplanar with its source parent and its signed-volume change is exactly zero.

Surrounding anatomy remains unfinished. An independent native012 step8000 census has exactly the same 4,007 raw boundary-crossing pairs as native009, including 1,177 against left pronator quadratus. Those contacts remain unclassified; passing child and joined-boundary checks does not waive them. Right forearm, the six-part back, final long baseline/intervention, and complete body qualification remain open. The failed compensated-position native011 was not promoted.

BodyParts3D anatomy is from the Database Center for Life Science under CC BY-SA 2.1 Japan; anatomy-source-attribution.json records member and archive identities. This increment changes triangulation within one source parent, following the existing conforming refinement and inferred passive deformation field. The body is an assembled mixed-source reference adult, not measurements of one person.
