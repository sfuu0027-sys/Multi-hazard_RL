from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "normalized_cases"
RESULT_GLOB = "cem_results_norm_case*.json"
RESULT_PREFIX = "cem_results_norm_"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _case_name(path: Path) -> str:
    stem = path.stem
    if not stem.startswith(RESULT_PREFIX):
        raise ValueError(f"Unexpected normalized result name: {path.name}")
    return stem[len(RESULT_PREFIX):]


def main() -> None:
    result_paths = sorted(ROOT.glob(RESULT_GLOB))
    if not result_paths:
        raise FileNotFoundError(f"No {RESULT_GLOB} files found.")

    generated: list[Path] = []
    for result_path in result_paths:
        result = _load_json(result_path)
        case_name = _case_name(result_path)
        lifecycle = deepcopy(result["lifecycle_config"])
        cem = deepcopy(result["cem_config"])
        if "iterations" not in cem and result.get("history"):
            cem["iterations"] = len(result["history"])
        cem.setdefault("plot_first_n", 30)
        cem.setdefault("plot_interval", 2)

        cfg = {
            "io": {
                "fig_dir": "fig_normalized",
                "out_json": f"cem_results_norm_{case_name}.json",
                "iter_trajectories": f"iter_trajectories_norm_{case_name}.json",
                "log_dir": "log_normalized",
            },
            "lifecycle": lifecycle,
            "cem": cem,
        }
        out_path = OUT_DIR / f"{case_name}_normalized.json"
        _write_json(out_path, cfg)
        generated.append(out_path)

    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
