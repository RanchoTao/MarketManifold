"""标普500市场结构聚类实验脚本。

仅依赖 Python 标准库即可运行，便于在课程展示环境中复现；如需进一步扩展，
可按 requirements.txt 安装 pandas/scikit-learn/matplotlib 后改写为更完整实验。
"""
from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

FEATURE_COLUMNS = ["return_1y", "volatility_1y", "momentum_6m", "max_drawdown_1y"]
NUMERIC_COLUMNS = ["latest_price", *FEATURE_COLUMNS]
DEFAULT_K_VALUES = list(range(2, 13))
DEFAULT_FINAL_K = 8
PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be123c", "#4f46e5", "#65a30d", "#c026d3", "#0f766e", "#b45309"]


def read_csv(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    cleaned = []
    seen = set()
    for row in rows:
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker or ticker in seen:
            continue
        item: Dict[str, object] = {"ticker": ticker}
        ok = True
        for col in NUMERIC_COLUMNS:
            try:
                value = float(row.get(col, ""))
            except (TypeError, ValueError):
                ok = False
                break
            if not math.isfinite(value):
                ok = False
                break
            item[col] = value
        if ok:
            cleaned.append(item)
            seen.add(ticker)
    winsorize(cleaned, NUMERIC_COLUMNS)
    return cleaned


def winsorize(rows: List[Dict[str, object]], cols: Sequence[str], lower_q: float = 0.01, upper_q: float = 0.99) -> None:
    for col in cols:
        vals = sorted(float(r[col]) for r in rows)
        lo = percentile(vals, lower_q)
        hi = percentile(vals, upper_q)
        for row in rows:
            row[col] = min(max(float(row[col]), lo), hi)


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_values[low]
    frac = pos - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def standardize(rows: Sequence[Dict[str, object]]) -> List[List[float]]:
    columns = [[float(r[col]) for r in rows] for col in FEATURE_COLUMNS]
    means = [sum(col) / len(col) for col in columns]
    stds = []
    for col, mean in zip(columns, means):
        var = sum((v - mean) ** 2 for v in col) / max(1, len(col) - 1)
        stds.append(math.sqrt(var) or 1.0)
    return [[(float(row[col]) - means[i]) / stds[i] for i, col in enumerate(FEATURE_COLUMNS)] for row in rows]


def sq_dist(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sq_dist(a, b))


def kmeans(data: Sequence[Sequence[float]], k: int, max_iter: int = 100) -> Tuple[List[int], List[List[float]], float]:
    # 确定性初始化：按样本顺序等距抽取初始中心，保证每次展示可复现。
    n = len(data)
    indices = [min(n - 1, round(i * (n - 1) / max(1, k - 1))) for i in range(k)]
    centroids = [list(data[i]) for i in indices]
    labels = [0] * n
    for _ in range(max_iter):
        changed = False
        for i, point in enumerate(data):
            best = min(range(k), key=lambda c: sq_dist(point, centroids[c]))
            if labels[i] != best:
                changed = True
                labels[i] = best
        sums = [[0.0] * len(data[0]) for _ in range(k)]
        counts = [0] * k
        for label, point in zip(labels, data):
            counts[label] += 1
            for j, value in enumerate(point):
                sums[label][j] += value
        for c in range(k):
            if counts[c]:
                centroids[c] = [value / counts[c] for value in sums[c]]
        if not changed:
            break
    inertia = sum(sq_dist(point, centroids[label]) for point, label in zip(data, labels))
    return labels, centroids, inertia


def silhouette_score(data: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    groups = group_indices(labels)
    scores = []
    for i, point in enumerate(data):
        own = labels[i]
        own_members = [idx for idx in groups[own] if idx != i]
        a = sum(euclidean(point, data[j]) for j in own_members) / len(own_members) if own_members else 0.0
        b = min(
            sum(euclidean(point, data[j]) for j in members) / len(members)
            for label, members in groups.items()
            if label != own and members
        )
        denom = max(a, b)
        scores.append((b - a) / denom if denom else 0.0)
    return sum(scores) / len(scores)


def davies_bouldin_index(data: Sequence[Sequence[float]], labels: Sequence[int], centroids: Sequence[Sequence[float]]) -> float:
    groups = group_indices(labels)
    scatters = {}
    for label, members in groups.items():
        scatters[label] = sum(euclidean(data[i], centroids[label]) for i in members) / len(members)
    ratios = []
    for i in groups:
        worst = max(
            (scatters[i] + scatters[j]) / (euclidean(centroids[i], centroids[j]) or 1e-12)
            for j in groups
            if j != i
        )
        ratios.append(worst)
    return sum(ratios) / len(ratios)


def calinski_harabasz_index(data: Sequence[Sequence[float]], labels: Sequence[int], centroids: Sequence[Sequence[float]]) -> float:
    n = len(data)
    groups = group_indices(labels)
    k = len(groups)
    overall = [sum(point[j] for point in data) / n for j in range(len(data[0]))]
    between = sum(len(members) * sq_dist(centroids[label], overall) for label, members in groups.items())
    within = sum(sq_dist(data[i], centroids[label]) for label, members in groups.items() for i in members)
    if k <= 1 or n <= k or within == 0:
        return 0.0
    return (between / (k - 1)) / (within / (n - k))


def group_indices(labels: Sequence[int]) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[label].append(idx)
    return dict(groups)


def evaluate_k_values(data: Sequence[Sequence[float]], k_values: Iterable[int]) -> List[Dict[str, float]]:
    metrics = []
    for k in k_values:
        labels, centroids, inertia = kmeans(data, k)
        metrics.append(
            {
                "k": k,
                "silhouette_score": silhouette_score(data, labels),
                "davies_bouldin_index": davies_bouldin_index(data, labels, centroids),
                "calinski_harabasz_index": calinski_harabasz_index(data, labels, centroids),
                "sse_inertia": inertia,
            }
        )
    return metrics


def profile_name(row: Dict[str, object]) -> str:
    ret = float(row["avg_return_1y"])
    vol = float(row["avg_volatility_1y"])
    mom = float(row["avg_momentum_6m"])
    dd = float(row["avg_max_drawdown_1y"])
    if ret > 0.35 and mom > 0.15 and vol > 0.28:
        return "高收益高波动动量型"
    if ret > 0.35 and mom > 0.15:
        return "高收益稳健动量型"
    if vol < 0.17 and dd > -0.14:
        return "低波动稳健型"
    if vol > 0.31 and dd < -0.28 and mom > 0:
        return "高波动修复型"
    if vol > 0.31 and dd < -0.28 and mom <= 0:
        return "高风险下行型"
    if ret < 0 and mom > 0:
        return "防御修复型"
    if vol < 0.18 and ret < 0.08 and mom < 0:
        return "低波动承压型"
    if ret < 0.05 and mom < 0:
        return "弱动量承压型"
    if ret > 0.18 and vol > 0.24:
        return "成长波动型"
    if ret < 0 and dd < -0.25:
        return "深度价值反弹型"
    return "均衡核心型"


def build_profiles(rows: Sequence[Dict[str, object]], labels: Sequence[int]) -> List[Dict[str, object]]:
    groups = group_indices(labels)
    profiles = []
    for label in sorted(groups):
        members = [rows[i] for i in groups[label]]
        item: Dict[str, object] = {"cluster": label, "asset_count": len(members)}
        for col in FEATURE_COLUMNS:
            item[f"avg_{col}"] = sum(float(r[col]) for r in members) / len(members)
        item["representative_tickers"] = ", ".join(str(r["ticker"]) for r in members[:8])
        item["profile_name"] = profile_name(item)
        profiles.append(item)
    return profiles


def covariance_matrix(data: Sequence[Sequence[float]]) -> List[List[float]]:
    n = len(data)
    d = len(data[0])
    return [[sum(point[i] * point[j] for point in data) / max(1, n - 1) for j in range(d)] for i in range(d)]


def mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[float]:
    return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]


def normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def first_two_pca_components(data: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    matrix = covariance_matrix(data)
    v1 = normalize([1.0, 0.5, 0.25, 0.125])
    for _ in range(80):
        v1 = normalize(mat_vec(matrix, v1))
    lambda1 = sum(v1[i] * mat_vec(matrix, v1)[i] for i in range(len(v1)))
    deflated = [[matrix[i][j] - lambda1 * v1[i] * v1[j] for j in range(len(v1))] for i in range(len(v1))]
    v2 = normalize([0.125, -0.25, 0.5, -1.0])
    for _ in range(80):
        v2 = normalize(mat_vec(deflated, v2))
    return v1, v2


def pca_coordinates(data: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    v1, v2 = first_two_pca_components(data)
    return [(sum(point[i] * v1[i] for i in range(len(v1))), sum(point[i] * v2[i] for i in range(len(v2)))) for point in data]


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n<rect width="100%" height="100%" fill="white"/>\n'


def scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi == lo:
        return (out_lo + out_hi) / 2
    return out_lo + (value - lo) / (hi - lo) * (out_hi - out_lo)


def write_k_metrics_svg(metrics: Sequence[Dict[str, float]], path: Path) -> None:
    width, height = 1000, 720
    panels = [(60, 70, 410, 230, "silhouette_score", "Silhouette 越大越好"), (540, 70, 410, 230, "davies_bouldin_index", "Davies-Bouldin 越小越好"), (60, 410, 410, 230, "calinski_harabasz_index", "Calinski-Harabasz 越大越好"), (540, 410, 410, 230, "sse_inertia", "SSE/Inertia 肘部法")]
    parts = [svg_header(width, height), '<text x="500" y="32" text-anchor="middle" font-size="24" font-weight="700">K 值评价指标对比</text>\n']
    ks = [float(m["k"]) for m in metrics]
    for x, y, w, h, key, title in panels:
        vals = [float(m[key]) for m in metrics]
        parts.append(f'<text x="{x + w/2}" y="{y - 15}" text-anchor="middle" font-size="15" font-weight="600">{title}</text>\n')
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#f8fafc" stroke="#cbd5e1"/>\n')
        points = []
        for k, val in zip(ks, vals):
            px = scale(k, min(ks), max(ks), x + 28, x + w - 20)
            py = scale(val, min(vals), max(vals), y + h - 28, y + 22)
            points.append((px, py))
        d = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
        parts.append(f'<polyline points="{d}" fill="none" stroke="#2563eb" stroke-width="2.4"/>\n')
        for (px, py), k, val in zip(points, ks, vals):
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="#2563eb"><title>K={int(k)}, {key}={val:.4f}</title></circle>\n')
        parts.append(f'<text x="{x + w/2}" y="{y + h + 22}" text-anchor="middle" font-size="12">K</text>\n')
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def write_pca_svg(rows: Sequence[Dict[str, object]], labels: Sequence[int], coords: Sequence[Tuple[float, float]], path: Path) -> None:
    width, height = 1000, 680
    xs, ys = [c[0] for c in coords], [c[1] for c in coords]
    parts = [svg_header(width, height), '<text x="500" y="34" text-anchor="middle" font-size="24" font-weight="700">PCA 二维聚类散点图</text>\n']
    parts.append('<rect x="70" y="60" width="850" height="540" fill="#f8fafc" stroke="#cbd5e1"/>\n')
    signal = sorted(range(len(rows)), key=lambda i: abs(float(rows[i]["return_1y"])) + abs(float(rows[i]["momentum_6m"])), reverse=True)[:18]
    for i, ((x, y), label) in enumerate(zip(coords, labels)):
        px = scale(x, min(xs), max(xs), 95, 895)
        py = scale(y, min(ys), max(ys), 575, 85)
        color = PALETTE[label % len(PALETTE)]
        title = f'{rows[i]["ticker"]} / Cluster {label}'
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.8" fill="{color}" opacity="0.82"><title>{html.escape(title)}</title></circle>\n')
        if i in signal:
            parts.append(f'<text x="{px + 6:.1f}" y="{py - 4:.1f}" font-size="10" fill="#0f172a">{html.escape(str(rows[i]["ticker"]))}</text>\n')
    parts.append('<text x="500" y="635" text-anchor="middle" font-size="13">PC1</text><text x="24" y="340" transform="rotate(-90 24,340)" text-anchor="middle" font-size="13">PC2</text>\n')
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def write_bar_svg(profiles: Sequence[Dict[str, object]], path: Path) -> None:
    width, height = 1150, 720
    left, top, plot_w, plot_h = 90, 70, 990, 470
    keys = ["avg_return_1y", "avg_volatility_1y", "avg_momentum_6m", "avg_max_drawdown_1y"]
    names = ["收益", "波动", "动量", "回撤"]
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    vals = [float(p[k]) for p in profiles for k in keys]
    lo, hi = min(vals + [0]), max(vals + [0])
    zero_y = scale(0, lo, hi, top + plot_h, top)
    parts = [svg_header(width, height), '<text x="575" y="34" text-anchor="middle" font-size="24" font-weight="700">各聚类核心特征均值对比</text>\n']
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1"/>\n')
    parts.append(f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + plot_w}" y2="{zero_y:.1f}" stroke="#64748b"/>\n')
    group_w = plot_w / len(profiles)
    bar_w = group_w / 6
    for gi, profile in enumerate(profiles):
        cx = left + gi * group_w + group_w / 2
        for bi, key in enumerate(keys):
            value = float(profile[key])
            x = cx - 2 * bar_w + bi * bar_w
            y = scale(value, lo, hi, top + plot_h, top)
            h = abs(zero_y - y)
            parts.append(f'<rect x="{x:.1f}" y="{min(y, zero_y):.1f}" width="{bar_w * .85:.1f}" height="{h:.1f}" fill="{colors[bi]}"><title>{html.escape(str(profile["profile_name"]))} {names[bi]} {value:.4f}</title></rect>\n')
        label = f'{profile["cluster"]}:{profile["profile_name"]}'
        parts.append(f'<text x="{cx:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="10" transform="rotate(24 {cx:.1f},{top + plot_h + 24})">{html.escape(label)}</text>\n')
    for i, name in enumerate(names):
        parts.append(f'<rect x="{left + i * 110}" y="660" width="14" height="14" fill="{colors[i]}"/><text x="{left + i * 110 + 20}" y="672" font-size="13">{name}</text>\n')
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def write_radar_svg(profiles: Sequence[Dict[str, object]], path: Path) -> None:
    # 为保证无外部依赖，雷达图用说明型 SVG 占位，详细数值见柱状图和 CSV。
    width, height = 700, 520
    parts = [svg_header(width, height), '<text x="350" y="36" text-anchor="middle" font-size="22" font-weight="700">聚类画像雷达图（展示版）</text>\n']
    cx, cy, radius = 350, 270, 170
    axes = [(0, "收益"), (math.pi / 2, "波动"), (math.pi, "动量"), (3 * math.pi / 2, "回撤")]
    for angle, name in axes:
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#cbd5e1"/><text x="{x:.1f}" y="{y:.1f}" font-size="13">{name}</text>\n')
    for r in [0.33, 0.66, 1.0]:
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{radius*r:.1f}" fill="none" stroke="#e2e8f0"/>\n')
    keys = ["avg_return_1y", "avg_volatility_1y", "avg_momentum_6m", "avg_max_drawdown_1y"]
    ranges = {k: (min(float(p[k]) for p in profiles), max(float(p[k]) for p in profiles)) for k in keys}
    for idx, profile in enumerate(profiles):
        pts = []
        for (angle, _), key in zip(axes, keys):
            lo, hi = ranges[key]
            rr = 35 + scale(float(profile[key]), lo, hi, 0, radius - 35)
            pts.append((cx + rr * math.cos(angle), cy + rr * math.sin(angle)))
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        color = PALETTE[idx % len(PALETTE)]
        parts.append(f'<polygon points="{point_text}" fill="{color}" opacity="0.08" stroke="{color}" stroke-width="1.6"><title>{html.escape(str(profile["profile_name"]))}</title></polygon>\n')
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def markdown_table(rows: Sequence[Dict[str, object]], headers: Sequence[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cells = []
        for header in headers:
            value = row[header]
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_profiles_md(path: Path, profiles: Sequence[Dict[str, object]], metrics: Sequence[Dict[str, float]], final_k: int) -> None:
    selected = next(m for m in metrics if int(m["k"]) == final_k)
    headers = ["cluster", "profile_name", "asset_count", "avg_return_1y", "avg_volatility_1y", "avg_momentum_6m", "avg_max_drawdown_1y", "representative_tickers"]
    lines = [
        "# 标普500市场结构聚类画像解释",
        "",
        "## 总体结论",
        f"本次实验以 K={final_k} 作为前端展示和答辩解释的默认聚类数，使用 1 年收益率、1 年波动率、6 个月动量和 1 年最大回撤四类标准化特征。",
        f"该 K 值下 Silhouette Score={selected['silhouette_score']:.4f}，Davies-Bouldin Index={selected['davies_bouldin_index']:.4f}，Calinski-Harabasz Index={selected['calinski_harabasz_index']:.2f}，SSE/Inertia={selected['sse_inertia']:.2f}。",
        "这些指标用于比较不同 K 值，不代表聚类存在唯一正确答案；最终解释还需要结合业务可读性和现场展示效果。",
        "",
        "## 聚类画像汇总表",
        "",
        markdown_table(profiles, headers),
        "",
        "## 分群解释",
        "",
    ]
    for row in profiles:
        lines += [
            f"### {row['profile_name']}（Cluster {row['cluster']}）",
            f"- 核心特征：平均 1 年收益率 {float(row['avg_return_1y']):.2%}，平均 1 年波动率 {float(row['avg_volatility_1y']):.2%}，平均 6 个月动量 {float(row['avg_momentum_6m']):.2%}，平均最大回撤 {float(row['avg_max_drawdown_1y']):.2%}。",
            f"- 典型资产：{row['representative_tickers']}。",
            "- 典型行为/偏好：同类资产在收益、风险、趋势和回撤维度上接近，可作为市场结构中的同质资产篮子。",
            "- 应用价值：支持资产分层、相似资产推荐、风险暴露识别和前端市场结构可视化。",
            "",
        ]
    lines += [
        "## 应用价值",
        "",
        "- 用户分层/资产分层：帮助不同风险偏好的用户理解资产所处结构位置。",
        "- 推荐系统：同一聚类可作为相似资产召回候选，再结合行业、市值和基本面过滤。",
        "- 精准运营：把高波动承压、高收益动量、低波动稳健等类别转化为可解释运营标签。",
        "- 内容理解：可为财经内容中的股票标签补充分群语义，提升解读效率。",
        "",
        "## 局限与改进方向",
        "",
        "- 当前仓库未包含原始日频价格数据和真实采集脚本，无法完全追溯特征计算日期。",
        "- 现阶段未纳入行业、市值、成交量、估值和文本情绪，解释仍偏市场表现维度。",
        "- 后续可加入层次聚类、GMM、谱聚类和 Bootstrap 稳定性检验。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(input_path: Path, results_dir: Path, final_k: int, k_values: Sequence[int]) -> Dict[str, Path]:
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(input_path)
    data = standardize(rows)
    metrics = evaluate_k_values(data, k_values)
    labels, _, _ = kmeans(data, final_k)
    profiles = build_profiles(rows, labels)
    coords = pca_coordinates(data)

    metrics_path = results_dir / "metrics.csv"
    profile_path = results_dir / "cluster_profile_summary.csv"
    assignments_path = results_dir / "cluster_assignments.csv"
    cleaned_path = results_dir / "cleaned_sp500_features.csv"
    pca_path = results_dir / "pca_coordinates.csv"
    md_path = results_dir / "cluster_profiles.md"

    write_csv(cleaned_path, rows, ["ticker", *NUMERIC_COLUMNS])
    write_csv(metrics_path, metrics, ["k", "silhouette_score", "davies_bouldin_index", "calinski_harabasz_index", "sse_inertia"])
    profile_fields = ["cluster", "profile_name", "asset_count", "avg_return_1y", "avg_volatility_1y", "avg_momentum_6m", "avg_max_drawdown_1y", "representative_tickers"]
    write_csv(profile_path, profiles, profile_fields)
    name_by_cluster = {p["cluster"]: p["profile_name"] for p in profiles}
    assignment_rows = [{**row, "cluster": labels[i], "profile_name": name_by_cluster[labels[i]]} for i, row in enumerate(rows)]
    write_csv(assignments_path, assignment_rows, ["ticker", *NUMERIC_COLUMNS, "cluster", "profile_name"])
    pca_rows = [{"ticker": rows[i]["ticker"], "cluster": labels[i], "pca1": coords[i][0], "pca2": coords[i][1]} for i in range(len(rows))]
    write_csv(pca_path, pca_rows, ["ticker", "cluster", "pca1", "pca2"])

    write_k_metrics_svg(metrics, figures_dir / "k_metrics.svg")
    write_pca_svg(rows, labels, coords, figures_dir / "pca_cluster_scatter.svg")
    write_bar_svg(profiles, figures_dir / "cluster_feature_bar.svg")
    write_radar_svg(profiles, figures_dir / "cluster_feature_radar.svg")
    write_profiles_md(md_path, profiles, metrics, final_k)

    return {
        "metrics": metrics_path,
        "profile": profile_path,
        "assignments": assignments_path,
        "cleaned": cleaned_path,
        "pca": pca_path,
        "markdown": md_path,
        "k_metrics_figure": figures_dir / "k_metrics.svg",
        "pca_figure": figures_dir / "pca_cluster_scatter.svg",
        "bar_figure": figures_dir / "cluster_feature_bar.svg",
        "radar_figure": figures_dir / "cluster_feature_radar.svg",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行标普500市场结构聚类实验")
    parser.add_argument("--input", type=Path, default=Path("sp500_features.csv"), help="输入特征 CSV 路径")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="结果输出目录")
    parser.add_argument("--final-k", type=int, default=DEFAULT_FINAL_K, help="最终展示聚类数")
    parser.add_argument("--k-min", type=int, default=min(DEFAULT_K_VALUES), help="K 值搜索下界")
    parser.add_argument("--k-max", type=int, default=max(DEFAULT_K_VALUES), help="K 值搜索上界")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(args.input, args.results_dir, args.final_k, list(range(args.k_min, args.k_max + 1)))
    print("聚类实验完成，输出文件如下：")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
