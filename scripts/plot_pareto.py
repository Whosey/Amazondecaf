import argparse
import csv
from collections import defaultdict
from pathlib import Path


COLORS = {
    "jo": "#2563eb",
    "so": "#16a34a",
    "fo": "#dc2626",
    "fen": "#9333ea",
    "soto": "#ea580c",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Render a DECAF CSV sweep as SVG.")
    parser.add_argument("--input", required=True, help="CSV produced by main.py sweep.")
    parser.add_argument("--output", required=True, help="Output SVG path.")
    return parser.parse_args()


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["utility_mean"] = float(row["utility_mean"])
        row["fairness_mean"] = float(row["fairness_mean"])
        row["beta"] = float(row["beta"])
        row["utility_sem"] = float(row.get("utility_sem") or 0.0)
        row["fairness_sem"] = float(row.get("fairness_sem") or 0.0)
    return rows


def pareto_front(rows):
    front = []
    for row in sorted(rows, key=lambda item: item["utility_mean"]):
        dominated = False
        for other in rows:
            no_worse = (
                other["utility_mean"] >= row["utility_mean"]
                and other["fairness_mean"] >= row["fairness_mean"]
            )
            strictly_better = (
                other["utility_mean"] > row["utility_mean"]
                or other["fairness_mean"] > row["fairness_mean"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(row)
    return front


def scale(value, src_min, src_max, dst_min, dst_max):
    if abs(src_max - src_min) < 1e-12:
        return (dst_min + dst_max) / 2.0
    frac = (value - src_min) / (src_max - src_min)
    return dst_min + frac * (dst_max - dst_min)


def render_svg(rows):
    width, height = 920, 620
    left, right, top, bottom = 86, 40, 50, 82
    plot_w = width - left - right
    plot_h = height - top - bottom

    utilities = [
        value
        for row in rows
        for value in (
            row["utility_mean"] - row["utility_sem"],
            row["utility_mean"] + row["utility_sem"],
        )
    ]
    fairnesses = [
        value
        for row in rows
        for value in (
            row["fairness_mean"] - row["fairness_sem"],
            row["fairness_mean"] + row["fairness_sem"],
        )
    ]
    x_min, x_max = min(utilities), max(utilities)
    y_min, y_max = min(fairnesses), max(fairnesses)
    x_pad = max(1e-6, (x_max - x_min) * 0.08)
    y_pad = max(1e-6, (y_max - y_min) * 0.08)
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def point(row):
        x = scale(row["utility_mean"], x_min, x_max, left, left + plot_w)
        y = scale(row["fairness_mean"], y_min, y_max, top + plot_h, top)
        return x, y

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="86" y="30" font-family="Arial" font-size="22" font-weight="700">DECAF Utility-Fairness Sweep</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>',
        f'<text x="{left + plot_w / 2}" y="{height - 28}" font-family="Arial" font-size="15" text-anchor="middle">System utility, higher is better</text>',
        f'<text x="24" y="{top + plot_h / 2}" font-family="Arial" font-size="15" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})">Fairness = -variance, higher is better</text>',
    ]

    for tick in range(6):
        x_value = x_min + tick * (x_max - x_min) / 5
        x = scale(x_value, x_min, x_max, left, left + plot_w)
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#111827"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" font-family="Arial" font-size="12" text-anchor="middle">{x_value:.2f}</text>')

        y_value = y_min + tick * (y_max - y_min) / 5
        y = scale(y_value, y_min, y_max, top + plot_h, top)
        parts.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#111827"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" font-family="Arial" font-size="12" text-anchor="end">{y_value:.2f}</text>')

    legend_x = left + plot_w - 160
    legend_y = top + 8
    for idx, method in enumerate(sorted(grouped)):
        color = COLORS.get(method, "#6b7280")
        y = legend_y + idx * 24
        parts.append(f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 14}" y="{y + 5}" font-family="Arial" font-size="14">{method.upper()}</text>')

    for method, method_rows in sorted(grouped.items()):
        color = COLORS.get(method, "#6b7280")
        front = pareto_front(method_rows)
        ordered = sorted(method_rows, key=lambda item: item["utility_mean"])
        if any(row["fairness_sem"] > 0 for row in ordered) and len(ordered) >= 2:
            upper = []
            lower = []
            for row in ordered:
                x = scale(row["utility_mean"], x_min, x_max, left, left + plot_w)
                y_upper = scale(
                    row["fairness_mean"] + row["fairness_sem"],
                    y_min,
                    y_max,
                    top + plot_h,
                    top,
                )
                y_lower = scale(
                    row["fairness_mean"] - row["fairness_sem"],
                    y_min,
                    y_max,
                    top + plot_h,
                    top,
                )
                upper.append((x, y_upper))
                lower.append((x, y_lower))
            polygon = " ".join(
                [f"{x:.2f},{y:.2f}" for x, y in upper]
                + [f"{x:.2f},{y:.2f}" for x, y in reversed(lower)]
            )
            parts.append(f'<polygon points="{polygon}" fill="{color}" opacity="0.12"/>')
        if len(front) >= 2:
            points = " ".join(f"{point(row)[0]:.2f},{point(row)[1]:.2f}" for row in front)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" opacity="0.85"/>')
        for row in method_rows:
            x, y = point(row)
            if row["utility_sem"] > 0:
                x0 = scale(row["utility_mean"] - row["utility_sem"], x_min, x_max, left, left + plot_w)
                x1 = scale(row["utility_mean"] + row["utility_sem"], x_min, x_max, left, left + plot_w)
                parts.append(f'<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x1:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="1.5" opacity="0.7"/>')
            if row["fairness_sem"] > 0:
                y0 = scale(row["fairness_mean"] - row["fairness_sem"], y_min, y_max, top + plot_h, top)
                y1 = scale(row["fairness_mean"] + row["fairness_sem"], y_min, y_max, top + plot_h, top)
                parts.append(f'<line x1="{x:.2f}" y1="{y0:.2f}" x2="{x:.2f}" y2="{y1:.2f}" stroke="{color}" stroke-width="1.5" opacity="0.7"/>')
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" fill="{color}" opacity="0.9"/>')
            parts.append(
                f'<text x="{x + 7:.2f}" y="{y - 7:.2f}" font-family="Arial" font-size="11" fill="#374151">b={row["beta"]:.2g}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def main():
    args = parse_args()
    rows = read_rows(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_svg(rows), encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
