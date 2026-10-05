# Two-scan CT vessel-mask profiles — 2026-10-05

This increment advances Numi Human's vascular geometry work from named visual surfaces to source-bound measurements of the Aorta and VCI label masks. It processes the original NIfTI masks for scans 001 and 002, records an area and RAS mask-centroid sample on every occupied source k-axis plane, and recomputes six-neighbour voxel-face connected components.

The final candidate is [`candidate.json`](candidate.json). The separate [`independent-verification.json`](independent-verification.json) rereads the registered archive and reproduces the per-plane areas, centroids, voxel totals, and six-neighbour component counts. It confirms four masks covering 280,824 voxels; each Aorta and VCI mask is one face-connected component with no empty source-k plane inside its occupied range.

| Scan | Label | Voxels | Occupied k planes | k range | Plane occupancy area (min / median / max, mm²) | 6-neighbour components |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 001 | Aorta | 124,336 | 155 | 488–642 | 4.77 / 557.90 / 2,486.23 | 1 |
| 001 | VCI | 27,457 | 102 | 486–587 | 58.17 / 256.06 / 543.59 | 1 |
| 002 | Aorta | 96,992 | 171 | 501–671 | 26.70 / 397.68 / 1,945.50 | 1 |
| 002 | VCI | 32,039 | 119 | 495–613 | 6.68 / 239.37 / 652.31 | 1 |

Areas are the exact occupied-voxel count times the NIfTI i-j parallelogram area. Centroids are arithmetic means of voxel-centre RAS coordinates within each plane. Face connectivity uses 4-connected components within each plane joined only where adjacent k planes share an occupied voxel.

These results characterize automatic scan masks. They do not verify that a label traces an open blood lumen instead of vessel wall or contrast region. The profiles follow the scanner's k-axis, not a reconstructed vessel path; they are not path-normal lumen areas or a medial-axis centreline. No vessel tree, pressure flow, blood mass, tissue exchange, subject binding, or physiology is admitted.

The local source archive contains only segmentation NIfTI files and the label-value workbook. It contains no DICOM or CT-intensity volume. TCIA's [collection page](https://www.cancerimagingarchive.net/collection/healthy-total-body-cts/) lists the segmentations as CC BY 4.0 and the 58.95 GB DICOM image collection under the NIH Controlled Data Access Policy; the next lumen-anatomy step therefore needs an access-cleared source image set or a separate openly accessible vessel-lumen reference. Dataset DOI: `10.7937/NC7Z-4F76`.

The candidate is bound to the exact archive, intake receipt, source config, NIfTI parser, and compiler hashes. The independent verifier requires NumPy and SciPy. Compilation, 12 focused regressions across the source importer, organ-volume compiler, and vessel-profile compiler, syntax checks, and the final independent replay passed in a local Python 3.11 environment using the declared volume dependencies. The project-local `numi` Python 3.14 runtime still lacks NumPy, so that installed alias was not used for this run. The first independent-verifier run stopped on a verifier-only 4D cross-product call before source comparison; the verifier was corrected to use the three spatial coordinates, and the subsequent complete independent replay passed. These toolchain details do not change the anatomy result or expand its evidence boundary.
