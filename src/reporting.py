from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import REPORTS_DIR, TABLES_DIR


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def write_structural_interpretation(metrics: pd.DataFrame) -> None:
    first = metrics.iloc[0]
    last = metrics.iloc[-1]
    lines = ["# Structural Interpretation", ""]
    lines.append("本分析仅为数据挖掘课程实验，不构成投资建议。")
    lines.append("")
    corr_change = last["mean_correlation"] - first["mean_correlation"]
    vol_change = last["market_volatility"] - first["market_volatility"]
    pc_change = last["pca_first_component_ratio"] - first["pca_first_component_ratio"]
    dist_change = last["mean_pairwise_distance"] - first["mean_pairwise_distance"]
    sector_change = last["sector_cluster_ari"] - first["sector_cluster_ari"]
    if corr_change > 0 and vol_change > 0 and pc_change > 0:
        label = "系统性联动增强"
    elif corr_change < 0 and dist_change > 0 and sector_change > 0:
        label = "市场分化增强"
    else:
        label = "结构变化混合，未形成单一方向判断"
    lines.extend([
        f"- 自动结构标签：**{label}**",
        f"- 平均相关性变化：{corr_change:.4f}",
        f"- 市场年化波动率变化：{vol_change:.4f}",
        f"- 第一主成分解释率变化：{pc_change:.4f}",
        f"- 平均二维距离变化：{dist_change:.4f}",
        f"- 行业与聚类 ARI 变化：{sector_change:.4f}",
        "",
        "上述标签只描述滚动窗口结构指标的变化，不代表确定性投资预测。",
    ])
    (TABLES_DIR / "structural_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


def write_report_tex(summary: dict | None = None, metrics: pd.DataFrame | None = None, prediction: pd.DataFrame | None = None) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = summary or {}
    has_results = metrics is not None and not metrics.empty
    if has_results:
        best_vol = metrics.loc[metrics["market_volatility"].idxmax()]
        result_text = (
            f"本次真实数据流程共保留 {summary.get('kept_ticker_count', '待生成')} 只股票，"
            f"共同交易日 {summary.get('final_trading_day_count', '待生成')} 个，"
            f"滚动窗口 {len(metrics)} 个。最高市场波动率窗口结束于 {best_vol['window_end']}。"
        )
    else:
        result_text = "尚未运行完整真实数据流程，结果待生成。"
    archive_text = latex_escape(summary.get("archive_path", "待生成"))
    source_text = latex_escape(summary.get("source", "unknown"))
    adjustment_text = latex_escape(summary.get("price_adjustment_status", "unknown"))
    tex = rf"""\documentclass[UTF8]{{ctexart}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{geometry}}
\usepackage{{hyperref}}
\geometry{{a4paper, margin=2.5cm}}
\title{{数据挖掘课程作业：近五年美国股票市场结构动态变化分析}}
\author{{MarketManifold}}
\date{{\today}}
\begin{{document}}
\maketitle
\begin{{abstract}}
本文基于本地压缩包中的真实美国股票日频历史数据，构建固定大市值股票池，使用滚动窗口相关距离、MDS、PCA、聚类与 Procrustes 对齐分析近五年市场结构动态变化。本分析仅为数据挖掘课程实验，不构成投资建议。
\end{{abstract}}

\section{{引言}}
股票市场在高波动时期往往表现出更强共同运动，但不同行业和个股之间也可能出现结构性分化。本文用数据驱动方式观察这种动态结构。

\section{{研究问题}}
本文关注收益率结构变化、低维空间迁移、平均相关性、市场波动率、共同因子强度、行业聚类关系以及下一窗口指标的简单预测。

\section{{数据来源}}
原始数据来自本地 ZIP：\texttt{{{archive_text}}}。压缩包来源判断为：\texttt{{{source_text}}}。由于文件未提供 adjusted close 字段，本文使用 \texttt{{close}}，复权状态记为 \texttt{{{adjustment_text}}}。

\section{{数据格式检查}}
数据检查输出见 \texttt{{../outputs/tables/archive\_inspection.json}} 和 \texttt{{../outputs/tables/archive\_sample\_report.md}}。检测到字段包括 ticker、period、date、time、open、high、low、close、volume、open interest，日期格式为 YYYYMMDD。

\section{{数据清洗与预处理}}
清洗步骤包括日期解析、股票代码标准化、非正价格过滤、重复日期保留最后一条、覆盖率筛选、最多 2 个交易日短缺口前向填充、共同交易日对齐、log return 计算以及 0.5\%/99.5\% winsorize。

\section{{滚动窗口设计}}
默认窗口长度为 90 个交易日，步长为 10 个交易日。每个窗口分别计算收益相关矩阵、相关距离矩阵、二维坐标、聚类标签和市场结构指标。

\section{{相关距离与 MDS}}
相关距离定义为 $d(i,j)=\sqrt{{2(1-\rho_{{ij}})}}$，并使用 MDS 投影到二维空间。二维坐标只表达相对结构，不代表固定经济坐标轴。

\section{{PCA 对照方法}}
对照方法将窗口内每只股票标准化收益序列作为特征，使用 PCA 降至二维，并用 KMeans 聚类与 MDS/Agglomerative 结果对照。

\section{{聚类方法}}
主方法使用 AgglomerativeClustering，对照方法使用 KMeans。聚类编号不被解释为固定行业。

\section{{Procrustes 动态对齐}}
第一个窗口作为基准，后续窗口相对前一个已对齐窗口做正交 Procrustes 对齐，以降低动画中的旋转、翻转和平移抖动。

\section{{市场结构指标}}
窗口指标包括平均相关性、中位相关性、等权市场年化波动率、横截面离散度、第一主成分解释率、聚类数量、轮廓系数、相邻窗口 ARI、行业-聚类 ARI 与平均两两距离。

\section{{实验结果}}
{result_text}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{../outputs/figures/market_metrics_timeseries.png}}
\caption{{滚动市场结构指标}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{../outputs/figures/latest_market_structure.png}}
\caption{{最新窗口市场结构}}
\end{{figure}}

\section{{关键时间窗口分析}}
关键窗口由真实指标自动选择，包括最高波动率、最高平均相关性、最低平均相关性或最新窗口。图见 \texttt{{../outputs/figures/key\_snapshot\_*.png}}。

\section{{动态结构变化分析}}
动态 GIF 输出为 \texttt{{../outputs/animations/market\_structure.gif}}。动画使用对齐后的 MDS 坐标，点大小代表窗口内股票波动率，颜色代表数据驱动聚类。

\section{{简单预测实验}}
预测实验按时间顺序划分训练集和测试集，比较 naive baseline、LinearRegression 和 RandomForestRegressor。结果见 \texttt{{../outputs/tables/prediction\_metrics.csv}}。

\section{{局限性}}
数据来源未被 ZIP 内说明文件独立确认；价格字段无 adjusted close 标识；股票池固定且有限；MDS 坐标存在低维投影损失；简单预测实验不能说明投资可交易性。

\section{{结论}}
MarketManifold 提供了一个可复现的课程实验流程，用真实历史数据展示美国股票市场结构在近五年滚动窗口中的聚集、分离和迁移。

\section{{参考文献}}
\begin{{enumerate}}
\item Mardia, Kent, and Bibby. Multivariate Analysis.
\item Hastie, Tibshirani, and Friedman. The Elements of Statistical Learning.
\item scikit-learn documentation.
\end{{enumerate}}
\end{{document}}
"""
    (REPORTS_DIR / "report.tex").write_text(tex, encoding="utf-8")
