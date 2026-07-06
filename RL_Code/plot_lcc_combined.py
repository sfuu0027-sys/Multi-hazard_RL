from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "lcc_rerun"
FIG_DIR = RESULT_ROOT / "fig"

TEX_BODY_FONT_PT = 11.0
TEX_TEXTWIDTH_IN = 6.5
TEX_HALF_WIDTH_IN = TEX_TEXTWIDTH_IN * 0.49
LINEWIDTH_SCALE = 0.6


BASE_CASES = [
    "case1_baseline",
    "case2a_fire_dominant",
    "case2b_eq_dominant",
    "case3a_cost_oriented",
    "case3b_risk_oriented",
    "case3c_resilience_oriented",
]

CASE4_CASES = [
    "case4a_no_resilience_equal",
    "case4b_no_resilience_risk_replacement",
    "case4c_no_risk_cost_emphasis",
]

CASE_ORDER = {
    "case1_baseline": 1,
    "case2a_fire_dominant": 2,
    "case2b_eq_dominant": 3,
    "case3a_cost_oriented": 4,
    "case3b_risk_oriented": 5,
    "case3c_resilience_oriented": 6,
    "case4a_no_resilience_equal": 7,
    "case4b_no_resilience_risk_replacement": 8,
    "case4c_no_risk_cost_emphasis": 9,
}

CASE_SHORT = {
    "case1_baseline": "case1",
    "case2a_fire_dominant": "case2a",
    "case2b_eq_dominant": "case2b",
    "case3a_cost_oriented": "case3a",
    "case3b_risk_oriented": "case3b",
    "case3c_resilience_oriented": "case3c",
    "case4a_no_resilience_equal": "case4a",
    "case4b_no_resilience_risk_replacement": "case4b",
    "case4c_no_risk_cost_emphasis": "case4c",
}

PALETTE = {
    "case1_baseline": "#ff8c00",
    "case2a_fire_dominant": "#0066ff",
    "case2b_eq_dominant": "#00cc44",
    "case3a_cost_oriented": "#ff00cc",
    "case3b_risk_oriented": "#ff3333",
    "case3c_resilience_oriented": "#00cfe6",
    "case4a_no_resilience_equal": "#7a5195",
    "case4b_no_resilience_risk_replacement": "#ef5675",
    "case4c_no_risk_cost_emphasis": "#ffa600",
}


def _lw(x: float) -> float:
    return float(x) * float(LINEWIDTH_SCALE)


def _tex_rc_params() -> dict[str, object]:
    base = TEX_BODY_FONT_PT
    tick = max(1.0, base - 1.0)
    legend = max(1.0, base - 1.0)
    return {
        "font.family": "Times New Roman",
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "legend.fontsize": legend,
        "figure.titlesize": base,
        "lines.linewidth": _lw(1.2),
        "savefig.dpi": 300,
    }


def _case_name_from_result(path: Path) -> str:
    prefix = "cem_results_"
    stem = path.stem
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected result name: {path.name}")
    return stem[len(prefix):]


def _moving_average(y: np.ndarray, window: int = 8) -> np.ndarray:
    w = int(max(1, window))
    if y.size < 2 or w <= 1:
        return y
    out = np.empty_like(y, dtype=float)
    for i in range(y.size):
        start = max(0, i - w + 1)
        out[i] = float(np.mean(y[start:i + 1]))
    return out


def _ewm_std(y: np.ndarray, alpha: float = 0.12) -> np.ndarray:
    a = float(min(0.95, max(0.05, alpha)))
    if y.size < 2:
        return np.zeros_like(y, dtype=float)
    mean = float(y[0])
    var = 0.0
    out = np.zeros_like(y, dtype=float)
    for i in range(1, y.size):
        x = float(y[i])
        prev_mean = mean
        mean = (1.0 - a) * mean + a * x
        var = (1.0 - a) * var + a * (x - prev_mean) * (x - mean)
        out[i] = math.sqrt(max(0.0, var))
    return out


def _load_histories(*, include_case4: bool = False) -> dict[str, list[dict[str, object]]]:
    histories: dict[str, list[dict[str, object]]] = {}
    for path in sorted(RESULT_ROOT.glob("case*/cem_results_case*.json"), key=lambda p: CASE_ORDER.get(_case_name_from_result(p), 999)):
        case_name = _case_name_from_result(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        histories[case_name] = data["history"]
    if include_case4:
        case4_dir = RESULT_ROOT / "case4_weight_reallocation" / "results"
        for path in sorted(case4_dir.glob("cem_results_case4*.json"), key=lambda p: CASE_ORDER.get(_case_name_from_result(p), 999)):
            case_name = _case_name_from_result(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            histories[case_name] = data["history"]
    expected = BASE_CASES + (CASE4_CASES if include_case4 else [])
    missing = [name for name in expected if name not in histories]
    if missing:
        raise FileNotFoundError(f"Missing LCC rerun histories: {missing}")
    return histories


def _series(history: list[dict[str, object]], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    iters = np.array([int(h["iteration"]) for h in history], dtype=float)
    if key == "loss":
        best_ret = np.array([float(h["best_of_iter"]["return"]) for h in history], dtype=float)
        elite_ret = np.array([float(h["elite_mean"]["return"]) for h in history], dtype=float)
        best = -best_ret
        elite = -elite_ret
        loss0 = float(best[0]) if best.size else 1.0
        if not np.isfinite(loss0) or loss0 <= 0.0:
            loss0 = 1.0
        loss_min = float(np.min(np.concatenate([best, elite])))
        denom = float(loss0 - loss_min)
        if not np.isfinite(denom) or abs(denom) <= 1.0e-12:
            denom = 1.0
        best = np.clip((best - loss_min) / denom, 0.0, None) * 1000.0
        elite = np.clip((elite - loss_min) / denom, 0.0, None) * 1000.0
        return iters, best, elite
    if key == "reward":
        best = np.array([float(h["best_of_iter"]["return"]) for h in history], dtype=float)
        elite = np.array([float(h["elite_mean"]["return"]) for h in history], dtype=float)
        return iters, best, elite
    metric_key = "lcc" if key == "cost" else key
    best = np.array([float(h["best_of_iter"][metric_key]) for h in history], dtype=float)
    elite = np.array([float(h["elite_mean"][metric_key]) for h in history], dtype=float)
    return iters, best, elite


def _apply_y_limits(ax: plt.Axes, key: str, all_y: list[float]) -> None:
    if not all_y:
        return
    ymax = float(np.nanmax(all_y))
    ymin = float(np.nanmin(all_y))
    if key == "cost":
        ax.set_ylim(top=max(140.0, ymax * 1.05))
    elif key == "risk":
        ax.set_ylim(bottom=min(-5.0, ymin * 1.05))
    elif key == "reward":
        pad = max(2.0, 0.08 * (ymax - ymin))
        ax.set_ylim(ymin - pad, ymax + pad)


def _draw_metric(
    ax: plt.Axes,
    histories: dict[str, list[dict[str, object]]],
    *,
    key: str,
    y_label: str,
    legend_loc: str,
    legend_ncols: int = 2,
) -> None:
    all_y: list[float] = []
    all_x: list[float] = []
    for case_name in sorted(histories, key=lambda n: CASE_ORDER.get(n, 999)):
        history = histories[case_name]
        iters, best_raw, elite_raw = _series(history, key)
        color = PALETTE[case_name]
        best_s = _moving_average(best_raw)
        elite_s = _moving_average(elite_raw)
        std_best = _ewm_std(best_raw)
        std_elite = _ewm_std(elite_raw)

        ax.plot(iters, best_raw, color=color, alpha=0.18, linewidth=_lw(1.2))
        ax.plot(iters, elite_raw, color=color, alpha=0.10, linewidth=_lw(1.0))
        ax.fill_between(iters, best_s - std_best, best_s + std_best,
                        color=color, alpha=0.10, linewidth=0.0)
        ax.fill_between(iters, elite_s - std_elite, elite_s + std_elite,
                        color=color, alpha=0.06, linewidth=0.0)
        ax.fill_between(iters, np.minimum(best_s, elite_s), np.maximum(best_s, elite_s),
                        color=color, alpha=0.12, linewidth=0.0)
        ax.plot(iters, elite_s, color=color, alpha=0.55, linewidth=_lw(1.6))
        ax.plot(iters, best_s, color=color, alpha=0.95,
                linewidth=_lw(2.0), label=CASE_SHORT[case_name])
        all_x.extend(iters.tolist())
        all_y.extend(best_raw.tolist())
        all_y.extend(elite_raw.tolist())

    ax.set_xlabel("Iteration")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(loc=legend_loc, ncols=legend_ncols, framealpha=0.9,
              fontsize=TEX_BODY_FONT_PT - 2.0, labelspacing=0.18,
              borderpad=0.25, handletextpad=0.5, columnspacing=0.8,
              handlelength=1.4, borderaxespad=0.25)
    xmax_data = max(all_x) if all_x else 1.0
    tick_step = 50.0 if xmax_data > 50.0 else max(1.0, math.ceil(xmax_data / 4.0))
    xmax = max(tick_step, math.ceil(xmax_data / tick_step) * tick_step)
    ax.set_xlim(0.0, xmax)
    ax.set_xticks(np.linspace(0.0, xmax, 5))
    _apply_y_limits(ax, key, all_y)


def _save_single(
    histories: dict[str, list[dict[str, object]]],
    *,
    key: str,
    y_label: str,
    out_path: Path,
    legend_loc: str,
    legend_ncols: int = 2,
) -> None:
    with plt.rc_context(_tex_rc_params()):
        fig, ax = plt.subplots(1, 1, figsize=(TEX_HALF_WIDTH_IN, TEX_HALF_WIDTH_IN * 0.72))
        _draw_metric(ax, histories, key=key, y_label=y_label,
                     legend_loc=legend_loc, legend_ncols=legend_ncols)
        fig.subplots_adjust(left=0.26, right=0.98, bottom=0.20, top=0.98)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)


def _save_five_panel(histories: dict[str, list[dict[str, object]]], out_path: Path) -> None:
    metrics = [
        ("loss", "Loss", "(a) Loss per step (normalised)", "upper right", 2),
        ("cost", "Cost", "(b) Cost per episode", "upper right", 2),
        ("lr", "Resilience loss", "(c) Resilience loss per episode", "upper right", 2),
        ("risk", "Risk", "(d) Risk per episode", "upper right", 2),
        ("reward", "Reward", "(e) Reward per episode", "lower right", 3),
    ]
    with plt.rc_context(_tex_rc_params()):
        fig = plt.figure(figsize=(TEX_TEXTWIDTH_IN, TEX_TEXTWIDTH_IN * 1.55))
        grid = gridspec.GridSpec(3, 4, figure=fig)
        axes = [
            fig.add_subplot(grid[0, 0:2]),
            fig.add_subplot(grid[0, 2:4]),
            fig.add_subplot(grid[1, 0:2]),
            fig.add_subplot(grid[1, 2:4]),
            fig.add_subplot(grid[2, 1:3]),
        ]
        for ax, (key, y_label, caption, legend_loc, legend_ncols) in zip(axes, metrics):
            _draw_metric(ax, histories, key=key, y_label=y_label,
                         legend_loc=legend_loc, legend_ncols=legend_ncols)
            ax.text(0.5, -0.28, caption, transform=ax.transAxes,
                    ha="center", va="top")
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.07, top=0.98,
                            wspace=0.48, hspace=0.88)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)


def _save_all_outputs(histories: dict[str, list[dict[str, object]]], *, prefix: str) -> None:
    legend_ncols = 3 if len(histories) > 6 else 2
    _save_single(histories, key="loss", y_label="Loss",
                 out_path=FIG_DIR / f"{prefix}_loss_per_step.png",
                 legend_loc="upper right", legend_ncols=legend_ncols)
    _save_single(histories, key="cost", y_label="Cost",
                 out_path=FIG_DIR / f"{prefix}_cost_per_episode.png",
                 legend_loc="upper right", legend_ncols=legend_ncols)
    _save_single(histories, key="lr", y_label="Resilience loss",
                 out_path=FIG_DIR / f"{prefix}_lor_per_episode.png",
                 legend_loc="upper right", legend_ncols=legend_ncols)
    _save_single(histories, key="risk", y_label="Risk",
                 out_path=FIG_DIR / f"{prefix}_risk_per_episode.png",
                 legend_loc="upper right", legend_ncols=legend_ncols)
    _save_single(histories, key="reward", y_label="Reward",
                 out_path=FIG_DIR / f"{prefix}_reward_per_episode.png",
                 legend_loc="lower right", legend_ncols=3)
    _save_five_panel(histories, FIG_DIR / f"{prefix}_combined_5panel.png")
    _save_five_panel(histories, FIG_DIR / f"{prefix}_combined.png")


def main() -> None:
    histories = _load_histories()
    histories_case1_4 = _load_histories(include_case4=True)
    _save_all_outputs(histories, prefix="rl_eval")
    _save_all_outputs(histories_case1_4, prefix="rl_eval_case1_4")
    print(FIG_DIR)


if __name__ == "__main__":
    main()
