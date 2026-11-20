# Skin interface volume trials

Fixed external and deep surfaces, connectivity and the first-order whole-material volume budget. The initial fitted patch and a patch grown through three connected interface rings both failed the local linear constraints for closing the measured 120/121 mass cut.

A maximum-capacity run on the three-ring patch tried the largest correction allowed by its linearized constraints. It reduced the selected volume by 2.159572e-7 m³ against a required 3.222657e-7 m³. Exact Float32 evaluation reported 1,373 inverted and 75 additional below-threshold tetrahedra. The candidate was rejected and no scene input changed. The verified zero-defect geometry is still the baseline.

Run the two fit scripts from the retained SSH Numi work directory with the paired-source masks and the previous capacity proof. The connected script used `--interface-rings 3` and distinct result prefixes `paired-interface-volume-connected` and `paired-interface-volume-capacity`. The maximum-capacity run also used `--maximum-capacity-only`. The linearized result is local to the selected patch and displacement budget.
