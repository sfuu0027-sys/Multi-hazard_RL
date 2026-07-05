from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "fig_normalized"
RESULT_GLOB = "cem_results_norm_case*.json"

TEX_BODY_FONT_PT = 11.0
TEX_TEXTWIDTH_IN = 6.5
TEX_HALF_WIDTH_IN = TEX_TEXTWIDTH_IN * 0.49
LINEWIDTH_SCALE = 0.6


CASE_ORDER = {
    "case1_baseline": 1,
    "case2a_fire_dominant": 2,
    "case2b_eq_dominant": 3,
    "case3a_cost_oriented": 4,
    "case3b_risk_oriented": 5,
    "case3c_resilience_oriented": 6,
}

CASE_SHORT = {
    "case1_baseline": "case1",
    "case2a_fire_dominant": "case2a",
    "case2b_eq_dominant": "case2b",
    "case3a_cost_oriented": "case3a",
    "case3b_risk_oriented": "case3b",
    "case3c_resilience_oriented": "case3c",
}

PALETTE = {
    "case1_baseline": "#ff8c00",
    "case2a_fire_dominant": "#0066ff",
    "case2b_eq_dominant": "#00cc44",
    "case3a_cost_oriented": "#ff00cc",
    "case3b_risk_oriented": "#ff3333",
    "case3c_resilience_oriented": "#00cfe6",
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
    prefix = "cem_results_norm_"
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


def _load_histories() -> dict[str, list[dict[str, object]]]:
    histories: dict[str, list[dict[str, object]]] = {}
    for path in sorted(ROOT.glob(RESULT_GLOB), key=lambda p: CASE_ORDER.get(_case_name_from_result(p), 999)):
        case_name = _case_name_from_result(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        histories[case_name] = data["history"]
    missing = [name for name in CASE_ORDER if name not in histories]
    if missing:
        raise FileNotFoundError(f"Missing normalized result histories: {missing}")
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
    best = np.array([float(h["best_of_iter"][key]) for h in history], dtype=float)
    elite = np.array([float(h["elite_mean"][key]) for h in history], dtype=float)
    return iters, best, elite


def _draw_metric(
    ax: plt.Axes,
    histories: dict[str, list[dict[str, object]]],
    *,
    key: str,
    y_label: str,
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
    ax.legend(loc="upper right", ncols=2, framealpha=0.9,
              fontsize=TEX_BODY_FONT_PT - 2.0, labelspacing=0.18,
              borderpad=0.25, handletextpad=0.5, columnspacing=0.8,
              handlelength=1.4, borderaxespad=0.25)
    xmax_data = max(all_x) if all_x else 1.0
    tick_step = 50.0 if xmax_data > 50.0 else max(1.0, math.ceil(xmax_data / 4.0))
    xmax = max(tick_step, math.ceil(xmax_data / tick_step) * tick_step)
    ax.set_xlim(0.0, xmax)
    ax.set_xticks(np.linspace(0.0, xmax, 5))
    if key in {"cost", "lcc"}:
        ax.set_ylim(top=max(140.0, float(np.nanmax(all_y)) * 1.05))
    elif key == "risk":
        ax.set_ylim(bottom=-5.0)


def _save_single(
    histories: dict[str, list[dict[str, object]]],
    *,
    key: str,
    y_label: str,
    out_path: Path,
) -> None:
    with plt.rc_context(_tex_rc_params()):
        fig, ax = plt.subplots(1, 1, figsize=(TEX_HALF_WIDTH_IN, TEX_HALF_WIDTH_IN * 0.72))
        _draw_metric(ax, histories, key=key, y_label=y_label)
        fig.subplots_adjust(left=0.26, right=0.98, bottom=0.20, top=0.98)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)


def _save_panel(histories: dict[str, list[dict[str, object]]], out_path: Path) -> None:
    metrics: list[tuple[str, str, str]] = [
        ("loss", "Loss", "(a) Loss per step (normalised)"),
        ("lcc", "Cost", "(b) Cost per episode"),
        ("lr", "Resilience loss", "(c) Resilience loss per episode"),
        ("risk", "Risk", "(d) Risk per episode"),
    ]
    with plt.rc_context(_tex_rc_params()):
        fig, axes = plt.subplots(2, 2, figsize=(TEX_TEXTWIDTH_IN, TEX_TEXTWIDTH_IN * 0.92))
        for ax, (key, y_label, caption) in zip(axes.flat, metrics):
            _draw_metric(ax, histories, key=key, y_label=y_label)
            ax.text(0.5, -0.32, caption, transform=ax.transAxes,
                    ha="center", va="top")
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.98,
                            wspace=0.34, hspace=0.66)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)


def main() -> None:
    histories = _load_histories()
    _save_single(histories, key="loss", y_label="Loss",
                 out_path=FIG_DIR / "rl_eval_loss_per_step.png")
    _save_single(histories, key="lcc", y_label="Cost",
                 out_path=FIG_DIR / "rl_eval_cost_per_episode.png")
    _save_single(histories, key="lr", y_label="Resilience loss",
                 out_path=FIG_DIR / "rl_eval_lor_per_episode.png")
    _save_single(histories, key="risk", y_label="Risk",
                 out_path=FIG_DIR / "rl_eval_risk_per_episode.png")
    _save_panel(histories, FIG_DIR / "rl_eval_combined_2x2.png")
    print(FIG_DIR)


if __name__ == "__main__":
    main()
