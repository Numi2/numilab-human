# Immutable mass ledger for conforming tissue refinement

The offline 534 helper composes source 002 edge subdivision with the frozen native parent 330 mass packet. For each refined cell it retains the original parent ID and typed material, geometric donor, anatomical region and stable source-face identity. The immutable ancestry measure is the parent 330 reference volume multiplied by the exact source subdivision fraction; operational source 002 geometry is never substituted for that measure.

The helper checks node barycentric reconstruction, ancestry containment in the original parent tetrahedron, matching geometric subdivision fractions, original-parent closure, duplicate and invalid topology, positive canonical Float32 cells and the existing material density bounds. It does not replace independent material-boundary intersection, anatomy or native physical checks.

## Actual SSH Numi source control

The actual 530 source control has 7,491 nodes and 29,642 cells. The existing native C++ mass assembler passes with 13.939678963460452 kg transferred, zero maximum per-parent mass/measure closure error and 148 source-donor groups closing within 1.11e-15 kg. Its 9,644 input source contribution rows produce 37,136 refined cell/source-donor entries. Child density remains within the existing skin and fat limits, ranging from 853.06 to 1098.61 kg/m3 across both materials.

Native negative controls reject replacement of immutable measure by operational geometry, cross-material ancestry and missing parent measure without mutating the output. Portable controls additionally cover invalid parent IDs, noninteger IDs, coordinate mismatch, nongeometric fractions, duplicate cells, ancestry outside its parent and invalid density.

Exact source, dependency, binary, command and ledger hashes are retained. The first checker attempt rejected a CRLF CSV header; its failure receipt is preserved, and the corrected checker accepts both newline conventions. The final helper revalidation reproduces the prepared arrays exactly and checks the authoritative mass CSV against its tetrahedron packet.

## Frame diagnosis and remaining admission

The proposed 029 anatomical fit incorrectly assigned skeleton-rest absolute coordinates to new FEM reference nodes. The historical NPZ current and reference arrays differ by at most 0.487 mm, while the wrong reference proposals jump 1.47–1.72 metres. Native import initializes both physical coordinate domains from canonical reference-world TetGen coordinates after checking the combined node map; the historical NPZ array ratio is not evidence of native prestress. The retained diagnostic identifies exact node/edge witnesses. Applying the intended current-pose displacement in the initialized reference frame removes that frame mismatch, but full target placement still creates invalid cells and density violations. A connected constrained fit remains necessary; this diagnostic is not a fitted candidate.

This increment covers the unprojected source refinement and mass ledger. It has no Runtime, GPU step, accepted physical step or motion qualification. The anatomical fit, fine-surface embedding, exact contact/nesting and native accepted motion remain pending. The existing accepted geometry, back repair, material laws and physical mass source are unchanged.
