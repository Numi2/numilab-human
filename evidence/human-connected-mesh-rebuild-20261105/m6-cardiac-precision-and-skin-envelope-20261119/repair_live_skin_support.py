"""Apply the native selector witness to the same registered skin candidate."""
from pathlib import Path
import csv
import json

import numpy as np

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/"resting-clearance-and-atrium-004"
def main():
    name="resting-support-selected-skin-witness.csv"
    runs=[HERE/f"selector-{key}"/"native"/name for key in ("control","candidate")]
    tables=[list(csv.DictReader(p.open())) for p in runs]
    assert all(len(x)>=624*32 for x in tables)
    control,candidate=next((a,b) for a,b in zip(*tables) if a!=b)
    k="selected_skin_vertex_local"
    starting=np.load(HERE/"skin-self-guarded.npz")["displacement_m"]
    selected=np.array(sorted({int(row[k]) for table in tables for row in table
                        if row[k].isdigit() and int(row[k])<len(starting)}),dtype=int)
    edited=selected[np.any(starting[selected]!=0.,axis=1)]
    assert int(candidate[k]) in edited
    displacement=starting.copy()
    displacement[edited]=0
    np.savez_compressed(HERE/"skin-live-pinned-proposal.npz",displacement_m=displacement)
    proof=dict(control=control,candidate=candidate,live_selected_indices=selected.tolist(),
               reverted_source_indices=edited.tolist(),retired_rows=len(edited),
               window_accepted_steps=[1,624],support_forces_unchanged_claim=False)
    (HERE/"skin-live-pinned-input.json").write_text(json.dumps(proof,indent=2)+"\n")
    print(f"Restored {len(edited)} source positions picked by the live contact reducer.")

if __name__=="__main__":
    main()
