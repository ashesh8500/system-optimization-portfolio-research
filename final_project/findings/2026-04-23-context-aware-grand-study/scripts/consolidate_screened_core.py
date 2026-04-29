from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "metrics"
FIGURES = ROOT / "figures"
SCREEN_ROOT = METRICS / "screened_candidate_pilot"
FIGURES.mkdir(parents=True, exist_ok=True)

UNIVERSES = ["liquid_us_equity_100", "liquid_us_equity_250", "liquid_us_equity_500"]


def _read_existing(name: str, filename: str) -> pd.DataFrame:
    path = SCREEN_ROOT / name / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["source_path"] = str(path.relative_to(ROOT))
    return df


def _safe_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.mean()) if not vals.empty else float("nan")


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    cumulative = (1.0 + returns.fillna(0.0)).cumprod()
    running_max = cumulative.cummax()
    dd = cumulative / running_max - 1.0
    return float(dd.min()) if not dd.empty else float("nan")


def main() -> None:
    summaries = []
    subperiods = []
    trials = []
    ledgers = []
    memberships = []
    for universe in UNIVERSES:
        summaries.append(_read_existing(universe, "summary.csv"))
        subperiods.append(_read_existing(universe, "subperiod_summary.csv"))
        trials.append(_read_existing(universe, "all_trials.csv"))
        ledgers.append(_read_existing(universe, "model_selection_ledger.csv"))
        memberships.append(_read_existing(universe, "screen_membership.csv"))

    summary = pd.concat([x for x in summaries if not x.empty], ignore_index=True)
    subperiod = pd.concat([x for x in subperiods if not x.empty], ignore_index=True)
    trial = pd.concat([x for x in trials if not x.empty], ignore_index=True)
    ledger = pd.concat([x for x in ledgers if not x.empty], ignore_index=True)
    membership = pd.concat([x for x in memberships if not x.empty], ignore_index=True)

    if summary.empty:
        raise SystemExit("No screened candidate summaries found.")

    # Standardize selected flags after combining all universes.
    summary["selected_for_paper"] = False
    idx = summary.groupby(["universe", "screen_rule"], dropna=False)["mean_sharpe"].idxmax()
    summary.loc[idx, "selected_for_paper"] = True
    summary = summary.sort_values(["universe", "screen_rule", "mean_sharpe"], ascending=[True, True, False])
    summary.to_csv(METRICS / "screened_core_summary.csv", index=False)

    # One winner per universe and per universe-screen pair.
    universe_winners = summary.loc[summary.groupby("universe")["mean_sharpe"].idxmax()].sort_values("mean_sharpe", ascending=False)
    universe_winners.to_csv(METRICS / "screened_universe_winners.csv", index=False)
    screen_winners = summary.loc[summary.groupby(["universe", "screen_rule"])["mean_sharpe"].idxmax()]
    screen_winners.to_csv(METRICS / "screened_universe_screen_winners.csv", index=False)

    # Ablation tables: screen contribution averaged across controllers, controller contribution averaged across screens.
    screen_ablation = (
        summary.groupby(["universe", "screen_rule"], as_index=False)
        .agg(
            mean_sharpe_across_controllers=("mean_sharpe", "mean"),
            best_sharpe=("mean_sharpe", "max"),
            mean_ann_return=("mean_ann_return", "mean"),
            mean_ann_vol=("mean_ann_vol", "mean"),
            worst_drawdown=("worst_split_drawdown", "min"),
            mean_turnover=("mean_turnover", "mean"),
            controller_count=("controller", "nunique"),
        )
        .sort_values(["universe", "mean_sharpe_across_controllers"], ascending=[True, False])
    )
    screen_ablation.to_csv(METRICS / "screen_ablation_summary.csv", index=False)

    controller_ablation = (
        summary.groupby(["universe", "controller"], as_index=False)
        .agg(
            mean_sharpe_across_screens=("mean_sharpe", "mean"),
            best_sharpe=("mean_sharpe", "max"),
            mean_ann_return=("mean_ann_return", "mean"),
            mean_ann_vol=("mean_ann_vol", "mean"),
            worst_drawdown=("worst_split_drawdown", "min"),
            mean_turnover=("mean_turnover", "mean"),
            screen_count=("screen_rule", "nunique"),
        )
        .sort_values(["universe", "mean_sharpe_across_screens"], ascending=[True, False])
    )
    controller_ablation.to_csv(METRICS / "controller_ablation_summary.csv", index=False)

    if not subperiod.empty:
        subperiod.to_csv(METRICS / "screened_core_subperiod_summary.csv", index=False)
        gross_net = (
            subperiod.groupby(["universe", "screen_rule", "controller", "cost_bps"], as_index=False)
            .agg(
                mean_ann_return=("ann_return", "mean"),
                mean_sharpe=("sharpe", "mean"),
                mean_turnover=("turnover", "mean"),
                worst_drawdown=("max_dd", "min"),
                splits=("split_id", "nunique"),
            )
            .sort_values(["universe", "screen_rule", "controller", "cost_bps"])
        )
        gross_net.to_csv(METRICS / "gross_vs_net_summary.csv", index=False)
        turnover = (
            subperiod.groupby(["universe", "screen_rule", "controller"], as_index=False)
            .agg(mean_turnover=("turnover", "mean"), median_turnover=("turnover", "median"), p90_turnover=("turnover", lambda x: float(np.nanpercentile(x, 90))))
        )
        turnover.to_csv(METRICS / "turnover_summary.csv", index=False)

    if not trial.empty:
        trial.to_csv(METRICS / "screened_core_all_trials.csv", index=False)
        # Multiple-testing diagnostic: number of tried arms and empirical winner gap by universe.
        mt_rows = []
        for universe, g in summary.groupby("universe"):
            sorted_g = g.sort_values("mean_sharpe", ascending=False)
            best = sorted_g.iloc[0]
            second = sorted_g.iloc[1] if len(sorted_g) > 1 else None
            mt_rows.append(
                {
                    "universe": universe,
                    "arms_tested": int(len(g)),
                    "screens_tested": int(g["screen_rule"].nunique()),
                    "controllers_tested": int(g["controller"].nunique()),
                    "best_screen_rule": best["screen_rule"],
                    "best_controller": best["controller"],
                    "best_mean_sharpe": float(best["mean_sharpe"]),
                    "second_best_mean_sharpe": float(second["mean_sharpe"]) if second is not None else np.nan,
                    "winner_gap_vs_second": float(best["mean_sharpe"] - second["mean_sharpe"]) if second is not None else np.nan,
                    "diagnostic_note": "Exploratory multiple-comparison count; no p-value claimed because folds are overlapping and arms are not independent.",
                }
            )
        pd.DataFrame(mt_rows).to_csv(METRICS / "multiple_testing_summary.csv", index=False)

        # Deflated Sharpe fallback: rank-based haircut by universe-arm count, transparent not formal DSR.
        ds_rows = []
        for universe, g in summary.groupby("universe"):
            n = len(g)
            penalty = np.sqrt(2.0 * np.log(max(n, 2))) / np.sqrt(max(float(g["splits"].median()), 1.0))
            for _, row in g.iterrows():
                ds_rows.append(
                    {
                        "universe": universe,
                        "screen_rule": row["screen_rule"],
                        "controller": row["controller"],
                        "mean_sharpe": row["mean_sharpe"],
                        "trial_count_penalty": penalty,
                        "haircut_sharpe": row["mean_sharpe"] - penalty,
                        "method_note": "Transparent exploratory Sharpe haircut; formal deflated Sharpe not claimed due overlapping walk-forward folds and non-independent arms.",
                    }
                )
        pd.DataFrame(ds_rows).to_csv(METRICS / "deflated_sharpe_summary.csv", index=False)

    if not ledger.empty:
        ledger.to_csv(METRICS / "screened_core_model_selection_ledger.csv", index=False)
    if not membership.empty:
        membership.to_csv(METRICS / "screened_core_membership.csv", index=False)
        concentration = (
            membership.groupby(["universe", "screen_rule", "symbol"], as_index=False)
            .agg(selection_count=("split_id", "nunique"), mean_rank=("rank", "mean"))
            .sort_values(["universe", "screen_rule", "selection_count", "mean_rank"], ascending=[True, True, False, True])
        )
        concentration.to_csv(METRICS / "screen_membership_concentration.csv", index=False)

    # Figures.
    import matplotlib.pyplot as plt

    pivot = summary.pivot_table(index="screen_rule", columns="universe", values="mean_sharpe", aggfunc="max")
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Best screened-controller Sharpe by universe and screen")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:.2f}" if pd.notna(val) else "", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Mean walk-forward Sharpe")
    fig.tight_layout()
    fig.savefig(FIGURES / "screened_core_sharpe_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for universe, g in summary.groupby("universe"):
        ax.scatter(g["mean_ann_vol"], g["mean_ann_return"], s=40 + 40 * g["mean_sharpe"].clip(lower=0), label=universe, alpha=0.75)
    ax.set_xlabel("Mean annualized volatility")
    ax.set_ylabel("Mean annualized return")
    ax.set_title("Screened core risk-return map")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "screened_core_risk_return_map.png", dpi=180)
    plt.close(fig)

    print("WROTE screened consolidation artifacts")
    print(universe_winners[["universe", "screen_rule", "controller", "mean_sharpe", "mean_ann_return", "worst_split_drawdown"]].to_string(index=False))


if __name__ == "__main__":
    main()
