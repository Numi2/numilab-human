"""Screen a registered adipose load path against its actual stress and geometry bounds."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from scipy import sparse


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def traction_bound(volume, gradients, fat_parents, shear, bulk, stretch, determinant):
    """Conservative bound on the total force on the fat parents, in any direction."""
    low, high = stretch
    j_min, j_max = determinant
    if not (0 < low <= 1 <= high and 0 < j_min <= 1 <= j_max and shear > 0 and bulk > 0):
        raise ValueError("Invalid material envelope")
    if gradients.shape[0] != 3 * len(volume) or np.any(volume <= 0):
        raise ValueError("Inconsistent integrated material geometry")
    # Virtual translation of all fat parents. The muscle parents stay where they are.
    grad = np.asarray(gradients[:, :fat_parents].sum(axis=1)).reshape(-1, 3)
    area = float(volume @ np.linalg.norm(grad, axis=1))
    # P = shear F + (bulk ln(J) - shear) F^-T. This bounds its largest
    # singular value for every orientation and every F in the declared envelope.
    stress = shear * high + max(abs(bulk * np.log(j_min) - shear),
                                 abs(bulk * np.log(j_max) - shear)) / low
    return {"effective_reference_area_m2": area, "piola_stress_upper_bound_Pa": float(stress),
            "total_force_upper_bound_N": area * float(stress)}


def assess(run, direction, target_N):
    cfg = json.loads((run / "launch.json").read_bytes())
    env = cfg["environment"]
    matrix = Path(env["NUMI_HUMAN_GLUTEAL_MATRIX"])
    if digest(matrix) != cfg["pins"][str(matrix)]:
        raise ValueError("Unmatched native stiffness")
    material = Path(env["NUMI_HUMAN_GLUTEAL_FAT_MASS_PREFLIGHT"]).parent / "material"
    with np.load(material / "parents.npz") as p:
        volume, nfat, donors = p["volume"], len(p["fat"]), p["donors"]
    grad = sparse.load_npz(material / "gradient.npz")
    with np.load(material / "reference-tangent.npz") as p:
        native_K = np.fromfile(matrix, "<f4").reshape(p["matrix"].shape)
        if not np.array_equal(p["matrix"].astype("f4"), native_K):
            raise ValueError("The native operator differs from the reconstructed material")
    shear, bulk = json.loads((material / "reference-tangent.json").read_bytes())["material_parameters_Pa"]
    K = native_K.astype(float)
    rigid = 6 * len(donors)
    if len(K) != 3 * nfat + rigid or abs(np.linalg.norm(direction)-1) > 1e-8:
        raise ValueError("Invalid body ports or bed normal")
    b = np.r_[np.tile(direction, nfat), np.zeros(rigid)]
    coeff = (b @ K)[:3*nfat].reshape(-1,3)
    slope = float(np.linalg.norm(coeff,axis=1).sum())
    # Independent norm balls on individual nodes give exactly this support function.
    optimum = coeff / np.maximum(np.linalg.norm(coeff,axis=1)[:,None], 1e-100)
    if not np.isclose(np.sum(coeff*optimum),slope):
        raise ValueError("Failed linear extremizer check")
    laws = {str(scale): traction_bound(volume,grad,nfat,shear*scale,bulk*scale,
                                        (2/3,1.5),(.4,2.5)) for scale in (.5,1.,1.5)}
    return {"matrix_sha256":digest(matrix),"gradient_sha256":digest(material/"gradient.npz"),
            "parents_sha256":digest(material/"parents.npz"),"volume_m3":float(volume.sum()),
            "elements":len(volume),"fat_parent_count":nfat,"donors":donors.tolist(),
            "shear_Pa":shear,"bulk_Pa":bulk,"stretch_bounds":[2/3,1.5],"volume_ratio_bounds":[.4,2.5],
            "native_linear_force_upper_bound_N_per_m_of_peak_relative_displacement":slope,
            "tested_three_site_share_N":target_N,"min_peak_relative_displacement_for_share_m":target_N/slope,
            "finite_strain_bounds_by_modulus_multiplier":laws,
            "scope":"Only this registered deep adipose compartment. The native linear bound fixes the relative bone pose. The finite-strain bound allows arbitrary nodal motion and admissible strains in the sourced nonlinear material. Other tissues can carry an independent share."}


if __name__ == "__main__":
    run,work=map(Path,sys.argv[1:])
    with np.load(work/"surface.npz") as q:normal=q["normal"]
    group=json.loads((work/"contact-regions.json").read_bytes())[1]["groups"]
    loaded=max(group,key=lambda g:g["normal_N"])
    report=assess(run,normal,loaded["normal_N_for_balanced_center_at_observed_sites"])
    (work/"gluteal-capacity.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report["finite_strain_bounds_by_modulus_multiplier"]["1.0"],report["min_peak_relative_displacement_for_share_m"])
