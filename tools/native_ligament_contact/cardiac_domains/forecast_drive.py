"""Use the calibrated Metal respiratory owner to propose cardiac proof domains."""

import json, os, subprocess, sys
from pathlib import Path
from certify_heart import sha


def run(source, cwd, case, out):
    settings = json.loads(source.read_text())
    native = json.loads((case / "launch.json").read_text())
    assert (
        native["environment"]["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"]
        == settings["environment"]["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"]
        == "1"
    )
    argv = settings["argv"][:]
    for path, digest in settings["pins"].items():
        if sha(Path(path)) != digest:
            raise ValueError(path)
    assert argv[4:] == ["--steps", "8000", "--dt", "0.002", "--vascular-dense45"]
    assert all(n in native["argv"] for n in argv[1:3])
    i = native["argv"].index("--resting-drive-intervention")
    argv[3] = str(out / "forecast.csv")
    argv[5] = "30000"
    argv += ["--drive-intervention", *native["argv"][i + 1 : i + 4]]
    out.mkdir(exist_ok=False)
    (out / "forecast-launch.json").write_text(
        json.dumps(
            dict(
                argv=argv,
                source=sha(source),
                native=sha(case / "launch.json"),
                pins=settings["pins"],
                scope="Authoring forecast; only the live native owner commits motion.",
            ),
            indent=2,
        )
        + "\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NM_", "NUMI_"))}
    subprocess.run(argv, cwd=cwd, env=env | settings["environment"], check=True)


if __name__ == "__main__":
    run(*(Path(p).resolve() for p in sys.argv[1:]))
