from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "lcc_rerun"
CONFIG_DIR = OUT_ROOT / "configs"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    case_paths = sorted(
        p for p in ROOT.glob("case*.json")
        if p.is_file() and p.stem.startswith("case")
    )
    if not case_paths:
        raise FileNotFoundError("No case*.json files found in RL_Code.")

    generated: list[Path] = []
    for src_path in case_paths:
        src = _load_json(src_path)
        case_name = src_path.stem
        cfg = deepcopy(src)
        cfg.setdefault("io", {})
        cfg.setdefault("lifecycle", {})

        cfg["io"]["fig_dir"] = f"lcc_rerun/{case_name}/fig"
        cfg["io"]["out_json"] = f"lcc_rerun/{case_name}/cem_results_{case_name}.json"
        cfg["io"]["iter_trajectories"] = f"lcc_rerun/{case_name}/iter_trajectories_{case_name}.json"
        cfg["io"]["log_dir"] = f"lcc_rerun/{case_name}/log"
        cfg["lifecycle"]["objective_cost_metric"] = "lcc"

        out_path = CONFIG_DIR / f"{case_name}.json"
        _write_json(out_path, cfg)
        generated.append(out_path)

    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
