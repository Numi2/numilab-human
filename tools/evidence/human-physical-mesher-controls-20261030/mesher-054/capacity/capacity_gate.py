from __future__ import annotations
from typing import Any

def evaluate_role_capacity(owner_result: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Check one material's generated node/cell counts; this is not admission."""
    required=("nodes","tetrahedra","positive_f32_cells")
    present=all(k in owner_result for k in required)
    max_nodes=profile.get("max_nodes_per_child")
    max_cells=profile.get("max_cells_per_child")
    limits_valid=(isinstance(max_nodes,int) and not isinstance(max_nodes,bool) and max_nodes>0 and
                  isinstance(max_cells,int) and not isinstance(max_cells,bool) and max_cells>0)
    try:
        nodes=int(owner_result["nodes"])
        cells=int(owner_result["tetrahedra"])
        positive=int(owner_result["positive_f32_cells"])
        counts_valid=(nodes>0 and cells>0 and positive>=0)
    except (KeyError,TypeError,ValueError,OverflowError):
        nodes=cells=positive=None
        counts_valid=False
    node_pass=bool(present and limits_valid and counts_valid and nodes<=max_nodes)
    cell_pass=bool(present and limits_valid and counts_valid and cells<=max_cells)
    owner_positive_status=(owner_result.get("status") == "POSITIVE_F32_CELLS_NOT_PHYSICAL_ADMISSION")
    positivity_pass=bool(present and counts_valid and owner_positive_status and positive==cells)
    return {
      "status":"PASS_PER_ROLE_COUNTS_ONLY" if node_pass and cell_pass and positivity_pass else "FAIL_PER_ROLE_CAPACITY_OR_POSITIVE_CELL_CHECK",
      "owner_positive_f32_status_pass":owner_positive_status,
      "profile_name":profile.get("name"),
      "scope":"one material-role output only",
      "nodes":nodes,"max_nodes_per_child":max_nodes,"node_limit_pass":node_pass,
      "tetrahedra":cells,"max_cells_per_child":max_cells,"cell_limit_pass":cell_pass,
      "positive_f32_cells":positive,"positive_count_matches_tetrahedra":positivity_pass,
      "combined_skin_fat_capacity":"NOT_EVALUATED",
      "physical_admission":False,"runtime":0,"accepted_steps":0,
    }
