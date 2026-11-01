"""Freeze and export simple candidates selected before Kaggle evaluation."""

import json, datetime
import numpy as np
import pandas as pd
from sweettv.simple_search import DEST, score, baseline
from sweettv.session_replay import cached as replay
from sweettv.experiment import Fold, ROOT
from sweettv.improve import WINDOWS
from sweettv.improve_final import export
from sweettv.blend import rrf
from threadpoolctl import threadpool_limits

with threadpool_limits(limits=1):
    selected = json.loads((DEST / "simple-selection.json").read_text())["selected"][
        "channel"
    ]
    rep = json.loads((DEST / "replay-selection.json").read_text())["selected"]
    (DEST / "submission-selection.json").write_text(
        json.dumps(
            {
                "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "channel": selected,
                "replay": rep,
                "rule": "Channel improves all 3 reused windows; replay improves two 5-week windows, long drops 0.00039. Both frozen before uploads; differences small, no significance claimed.",
            },
            indent=2,
        )
    )
    f = Fold(*WINDOWS["final"])
    b = rrf(f, baseline("final"))
    results = []
    for name, r, s in [
        ("channel", selected, score(f, selected["config"])),
        ("replay", rep, replay("final", rep["days"], rep["weekly"])[rep["mode"]]),
    ]:
        z = (1 - r["weight"]) * b + r["weight"] * rrf(f, s)
        np.save(DEST / f"final_{name}_blend.npy", z)
        v = export(f, z, f"v5_{name}_blend")
        csv = pd.read_csv(ROOT / v["file"], dtype=str)
        allowed = set(
            pd.read_csv(
                ROOT / "data/raw/export_arh_31-42-final.csv", usecols=["tv_show_id"]
            ).tv_show_id
        ) - {0}
        assert csv.user_id.tolist() == f.template.user_id.tolist()
        assert all(
            len(set(x.split())) == len(x.split()) == 5
            and set(map(int, x.split())) <= allowed
            for x in csv.tv_show_id
        )
        v.update(config=r, raw_schedule_verified=True)
        results.append(v)
    (DEST / "exports.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results), flush=True)
