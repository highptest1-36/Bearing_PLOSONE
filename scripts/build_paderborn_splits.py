"""Write frozen Paderborn bearing-disjoint split manifests (protocol V2 §4.2-4.4) with content hashes."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pave.data.paderborn import BEARINGS  # noqa: E402
from pave.experiments.phase2 import PU_A2R, PU_CHALLENGE, check_bearing_disjoint, pu_bd5_sets  # noqa: E402

OUT = ROOT / "03_MANIFESTS" / "paderborn" / "splits"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in pu_bd5_sets() + [PU_A2R]:
        check_bearing_disjoint(s)
        proto = "PU_A2R" if s["fold"] == "A2R" else "PU_BD5"
        assign = [{"bearing": b, "role": r, "label3": BEARINGS[b]["label3"], "group": BEARINGS[b]["group"]}
                  for r in ("train", "val", "test") for b in sorted(s[r])]
        h = hashlib.sha256(json.dumps(assign, sort_keys=True).encode()).hexdigest()
        d = {"protocol": proto, "fold_id": s["fold"], "assignments": assign, "content_sha256": h,
             "challenge_descriptive_only": sorted(PU_CHALLENGE),
             "notes": ["All 4 operating settings and all measurements of a bearing share one role.",
                       "Bearing-disjoint: no bearing appears in two roles.",
                       "Source of PU_BD5 rows: Lessmeier et al. 2016, Table 10."]}
        (OUT / f"split_{proto}_{s['fold']}.json").write_text(json.dumps(d, indent=2))
        rows.append(f"{proto},{s['fold']},{h},{len(s['train'])},{len(s['val'])},{len(s['test'])}")
    (OUT / "split_index.csv").write_text("protocol,fold_id,content_sha256,n_train_bearings,n_val_bearings,n_test_bearings\n"
                                         + "\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
