from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "Experiments"
CONSTRAINED = EXPERIMENTS / "Constrained"
GROUP = "Children"
POF_LIMIT = 1.05


def _to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = [[str(row[col]) for col in columns] for _, row in df.iterrows()]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _best_under_pof(df: pd.DataFrame, baseline_total: float, baseline_gini: float) -> pd.Series:
    work = df.copy()
    work["PoF"] = work["Total"] / baseline_total
    work["BoF_gini"] = 1.0 - (work["Gini"] / baseline_gini)
    feasible = work[work["PoF"] <= POF_LIMIT]
    if feasible.empty:
        raise RuntimeError(f"No feasible row under PoF <= {POF_LIMIT}")
    return feasible.sort_values(["BoF_gini", "PoF"], ascending=[False, True]).iloc[0]


def main() -> None:
    official = pd.read_csv(EXPERIMENTS / "GIFF_vs_SI-X.csv")
    official_row = official[official["Group"] == GROUP].iloc[0]

    giff = pd.read_csv(CONSTRAINED / "GIFF.csv")
    six = pd.read_csv(CONSTRAINED / "SI-X.csv")
    giff = giff[giff["Group"] == GROUP]
    six = six[six["Group"] == GROUP]

    baseline_total = float(giff[giff["Beta"] == 0]["Total"].iloc[0])
    baseline_gini = float(giff[giff["Beta"] == 0]["Gini"].iloc[0])

    giff_best = _best_under_pof(giff, baseline_total, baseline_gini)
    six_best = _best_under_pof(six, baseline_total, baseline_gini)

    summary = pd.DataFrame(
        [
            {
                "Group": GROUP,
                "Source": "official_reported",
                "Baseline_Total": official_row["Baseline_Total"],
                "Baseline_Gini": official_row["Baseline_Gini"],
                "GIFF_beta": official_row["GIFF_beta"],
                "GIFF_BoF_gini": official_row["GIFF_BoF_gini"],
                "GIFF_PoF": official_row["GIFF_PoF"],
                "SI-X_beta": official_row["SI-X_beta"],
                "SI-X_BoF_gini": official_row["SI-X_BoF_gini"],
                "SI-X_PoF": official_row["SI-X_PoF"],
                "Best_Method": official_row["Best_Method"],
            },
            {
                "Group": GROUP,
                "Source": "local_reproduction",
                "Baseline_Total": baseline_total,
                "Baseline_Gini": baseline_gini,
                "GIFF_beta": giff_best["Beta"],
                "GIFF_BoF_gini": giff_best["BoF_gini"],
                "GIFF_PoF": giff_best["PoF"],
                "SI-X_beta": six_best["Beta"],
                "SI-X_BoF_gini": six_best["BoF_gini"],
                "SI-X_PoF": six_best["PoF"],
                "Best_Method": "GIFF" if giff_best["BoF_gini"] >= six_best["BoF_gini"] else "SI-X",
            },
        ]
    )

    out_csv = CONSTRAINED / "children_reproduction_summary.csv"
    out_md = CONSTRAINED / "children_reproduction_summary.md"
    summary.to_csv(out_csv, index=False)
    out_md.write_text(_to_markdown(summary), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
