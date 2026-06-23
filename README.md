# MarketManifold

数据挖掘课程作业：近五年美国股票市场结构动态变化分析。

本项目从本地 `d_us_txt.zip` 真实历史数据中读取固定美国大市值股票池，构建清洗后的价格矩阵和收益率矩阵，并用滚动窗口分析市场相关结构、低维聚集迁移、聚类稳定性、行业与数据驱动聚类关系，以及下一窗口市场指标的简单预测。

## 研究问题

- 近五年美国股票收益率结构如何变化；
- 每个滚动窗口内股票在低维空间中的聚集、分离和迁移；
- 市场平均相关性、整体波动率和共同因子强度如何变化；
- 高波动时期市场结构是否更集中；
- 行业聚类与数据驱动聚类之间有什么关系；
- 是否可以用简单模型预测下一窗口平均相关性或波动率。

## 数据说明

将原始 ZIP 放在项目根目录，默认文件名为 `d_us_txt.zip`。项目不会完整解压 ZIP，而是用 Python 标准库 `zipfile` 流式读取目标股票文件。

数据来源只写作 `Stooq-style (not independently confirmed)`：目录结构、文件名和字段格式呈现 Stooq 风格，但 ZIP 内没有独立说明文件，因此不能写成已确认的 Stooq 数据源。

原始 ZIP、缓存和大体积临时文件不应上传 GitHub，因为它们体积大且可以由脚本重新生成。本仓库应保留代码、报告、较小结果表和图表。

## Windows 安装与运行

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py
```

## 常用命令

检查压缩包：

```powershell
python scripts/inspect_archive.py
python run_pipeline.py --inspect-only
```

只清洗数据：

```powershell
python scripts/prepare_data.py
python run_pipeline.py --prepare-only
```

一键运行：

```powershell
python run_pipeline.py --archive d_us_txt.zip --years 5 --window 90 --step 10 --clusters 6
```

只运行分析：

```powershell
python run_pipeline.py --analysis-only
```

测试：

```powershell
python -m pytest -q
```

## 方法

主方法在每个滚动窗口中计算股票收益相关矩阵，并转换为相关距离：

```text
d(i,j) = sqrt(2 * (1 - corr(i,j)))
```

随后使用 MDS 降到二维，并使用 AgglomerativeClustering 聚类。对照方法将窗口内每只股票的标准化收益序列作为特征，使用 PCA 降到二维，并使用 KMeans 聚类。

不同窗口独立降维会产生任意旋转、翻转和平移。项目对相邻窗口坐标执行正交 Procrustes 对齐，动画使用对齐后的坐标。二维坐标只表达相对结构，不代表固定经济坐标轴。

## 输出文件

- `outputs/tables/archive_inspection.json`：ZIP 结构检查；
- `outputs/tables/archive_sample_report.md`：样本文件报告；
- `outputs/tables/ticker_mapping_report.csv`：股票匹配报告；
- `data/processed/prices_long.csv`：长表价格；
- `data/processed/prices_wide.csv`：宽表价格；
- `data/processed/log_returns.csv`：log return；
- `outputs/tables/data_summary.json`：数据摘要；
- `outputs/tables/data_quality_report.csv`：数据质量；
- `outputs/tables/window_metrics.csv`：滚动窗口指标；
- `outputs/tables/window_coordinates.csv`：窗口坐标；
- `outputs/tables/window_clusters.csv`：聚类标签；
- `outputs/figures/*.png`：静态图；
- `outputs/animations/market_structure.gif`：动态结构动画；
- `reports/report.tex`：课程报告。

## 报告编译

报告使用 XeLaTeX 编译中文：

```powershell
cd reports
xelatex -interaction=nonstopmode report.tex
```

如果本机没有 `xelatex`，pipeline 会记录状态，但不会把它视为致命错误。

## 常见错误

- 找不到 ZIP：将 `d_us_txt.zip` 放到项目根目录，或使用 `--archive` 指定路径；
- 匹配股票少于 30 只：查看 `outputs/tables/ticker_mapping_report.csv`；
- 没有标准化数据却运行 `--analysis-only`：先运行 `python scripts/prepare_data.py`；
- 没有 MP4：安装 ffmpeg，或使用已生成的 GIF。

## 数据限制与免责声明

如果 ZIP 内没有 adjusted close 字段或说明文件，本项目不能声称价格为复权价格，会把复权状态记为 `unknown`，并标记单日绝对收益超过 50% 的异常跳变。

本分析仅为数据挖掘课程实验，不构成投资建议。

## GitHub 上传建议

应提交：代码、`README.md`、`requirements.txt`、`reports/report.tex`、较小的 `outputs/tables`、必要图表和测试。

不应提交：`d_us_txt.zip`、任何 `*.zip`、`data/raw/`、`data/cache/`、大型动画 MP4、虚拟环境和缓存。

首次提交示例：

```powershell
git init
git add .gitignore README.md requirements.txt run_pipeline.py scripts src tests reports notebooks outputs/tables outputs/figures outputs/animations/market_structure.gif
git commit -m "Build MarketManifold course project pipeline"
```
