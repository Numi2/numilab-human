"""Read FEBio's text observations without turning archived output into a new run."""
from __future__ import annotations

import math
import re


FIELDS = {
    "center_of_mass": (3, "mm"),
    "rotation_quaternion": (4, "dimensionless; xyzw"),
    "Reaction_Forces": (3, "N"),
    "Reaction_Torques": (3, "N mm"),
    "Rigid_Connector_Force": (3, "N"),
    "Rigid_Connector_Moment": (3, "N mm"),
}
RIGID_BODY_IDS = {1, 2, 3, 4, 17, 18, 19, 20, 21}
CONNECTOR_IDS = set(range(1, 9))


def parse_febio_log(text: str) -> dict:
    """Retain accepted observations and trial failures as different evidence.

    Units here are the pinned oks003 mm/N source convention, not a convention
    inferred from arbitrary FEBio logs. Continuation time is not physical time.
    Missing/truncated records never count as successful completion.
    """
    compact = re.sub(r"[ \t]", "", text)
    version = re.search(r"version-(\d+\.\d+\.\d+)", compact)
    converged = [float(x) for x in re.findall(r"converged at time\s*:\s*([^\s]+)", text)]
    if any(not math.isfinite(x) for x in converged):
        raise ValueError("nonfinite accepted source time")
    if any(a >= b for a, b in zip(converged, converged[1:])):
        raise ValueError("nonmonotone accepted source times")
    records = []
    seen = set()
    pattern = r"Data Record #(\d+)\s*\n=+\s*\nStep = (\d+)\s*\nTime = ([^\s]+)\s*\nData = ([^\n]+)\n((?:[^\S\n]*\d+[^\n]*\n)+)"
    for match in re.finditer(pattern, text.replace("\r\n", "\n")):
        number, step, time, field, rows = match.groups()
        field = field.strip()
        if field not in FIELDS:
            raise ValueError(f"unsupported source log observation: {field}")
        key = int(step), field
        if key in seen:
            raise ValueError(f"duplicate source observation: {key}")
        seen.add(key)
        values = {}
        for row in rows.splitlines():
            columns = row.split()
            identifier, vector = int(columns[0]), [float(x) for x in columns[1:]]
            if identifier in values or len(vector) != FIELDS[field][0] or not all(map(math.isfinite, vector)):
                raise ValueError(f"invalid source observation row: {field}")
            values[identifier] = vector
        time = float(time)
        if not math.isfinite(time):
            raise ValueError("nonfinite source observation time")
        records.append({"record": int(number), "step": int(step), "continuation_time": time,
                        "field": field, "units": FIELDS[field][1], "values": values})
    declared = len(re.findall(r"Data Record #", text))
    complete = declared == len(records)
    steps = sorted({r["step"] for r in records})
    for step in steps:
        group = [r for r in records if r["step"] == step]
        complete &= {r["field"] for r in group} == set(FIELDS)
        complete &= len({r["continuation_time"] for r in group}) == 1
        complete &= all(set(r["values"]) == (CONNECTOR_IDS if r["field"].startswith("Rigid_Connector")
                                              else RIGID_BODY_IDS) for r in group)
        if step < 1 or step > len(converged):
            complete = False
        else:
            # FEBio prints the convergence banner with %g (six significant
            # digits), but the data-record time has more digits. Compare at
            # the banner's precision; do not change the observation values.
            complete &= all(format(r["continuation_time"], ".6g") == format(converged[step-1], ".6g")
                            for r in group)
    complete &= steps == list(range(1, len(converged)+1)) and bool(steps)
    normal = "NORMALTERMINATION" in compact
    failed = "ERRORTERMINATION" in compact
    count = re.search(r"Number of time steps completed[^:]*:\s*(\d+)", text)
    complete &= count is not None and int(count[1]) == len(converged)
    # Terminal footer alone is insufficient; retain all trial errors separately.
    status = ("normal_termination_with_complete_observations" if normal and not failed and complete
              else "failed_or_incomplete")
    return {"status": status, "version": version[1] if version else None,
            "time_semantics": "quasi-static continuation parameter, not elapsed physical seconds",
            "normal_termination_reported": normal, "error_termination_reported": failed,
            "observations_complete": bool(complete), "accepted_times": converged,
            "accepted_increment_count": len(converged),
            "negative_jacobian_trial_diagnostics": len(re.findall("Negative jacobian was detected", text)),
            "warning_blocks": len(re.findall(r"\*\s+WARNING\s+\*", text)),
            "error_blocks": len(re.findall(r"\*\s+ERROR\s+\*", text)),
            "records": records}
