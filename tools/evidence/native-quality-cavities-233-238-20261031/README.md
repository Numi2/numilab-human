# Exact native-quality cavity preparation

These tools improve offline Skin/Fat compiler candidates using the existing native quality and exact Float32 geometry owners. They do not advance physical state, allocate mass, or launch the native simulator. Every result remains physical_admission=false and native_steps=0.

The 233 solver retains the original strict native-quality path by default. An explicit construction budget permits positive tetrahedra that have not yet met the final native quality gate, with an integer constraint on their count. This lets a candidate improve incrementally rather than requiring every local cavity to become fully qualified in one solve. The native threshold and admission code are unchanged. Five controls check the strict default, explicit positive defect budget, zero-budget rejection, invalid budget rejection and exact positive default result.

234/235 require strict total native-defect descent for each replacement. They retain exact coordinates, material identity, directed cavity boundary and exact Float32 volume, and use the existing exact internal-face intersection owner. 235 caches only solver-infeasible identical cavities with coordinate, cell-set, material, budget and owner hashes; changed coordinates, budget or connectivity invalidate the key. Seven controls cover this cache and the actual cavity call. The second cached body batch reused 7,535 unchanged failed cavities. 237 extends the same method to pairs of neighboring vertex cavities.

236/238 replay only exact matching material/cell tuples onto the independently repaired Skin boundary. Missing or conflicting source cavities are skipped with witnesses. Their full-body checks retain positive cell signs, unchanged directed boundary, duplicate rejection and exact native-defect accounting. 238 also carries the numerical cavity cache as internal compiler metadata.

The passing join in joined-volume-quality-238-root/skin224-through009-run001 has 718,468 cells, zero nonpositive cells, and 16,299 remaining native-quality failures. It contains the 11-facet DP boundary repair and all compatible quality increments; eight desired boundary facets remain missing and twenty extra facets remain. The earlier 236 joins record three conflicting patches that were skipped. This is one integrated preparation candidate, not an admitted native scene.

The manifests pin the executed sources and original full reports byte for byte. Geometry payloads and per-cavity heavy outputs remain at their pinned private workspace paths. Imports reuse the existing 198, 209, 218 and 349 owners in that workspace. The preceding source snapshot is in commit 998522387fa2061a5440e99ee9429e5239a97396.

A complete matching surface/volume, final zero native-quality defects, conservative source mass and moments, regenerated registered transfer, and native accepted-motion checks remain required. No performance, physiological or motion qualification follows from these compiler timings.
