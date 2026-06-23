# 基于动态网络与聚类分析的标普500市场结构演化研究

本项目是数据挖掘课程项目 **RanchoTao/Manifold** 的实现版。项目已经从“项目提案”推进到“可展示、可复现、可答辩”的实验闭环：以标普500股票价格面板为基础，把连续滚动窗口构造成市场快照，对市场快照进行无监督聚类，识别不同的市场结构状态（Market Regime），并解释这些状态如何随重大事件演化。

> 当前仓库采用“真实数据优先、可复现模拟数据兜底”的策略：如果 `data/raw/sp500_prices.csv` 存在，脚本会读取真实价格面板；如果不存在，会自动生成一个结构清晰、可复现、包含 COVID 冲击、加息周期和 AI 行情等阶段的模拟价格面板，保证课程展示环境中可以从头运行。

## 1. 当前仓库审查结论

| 模块 | 当前状态 | 说明 |
| --- | --- | --- |
| 项目主题 | 已明确 | 研究对象是“滚动窗口市场快照”的结构状态，不是简单的静态股票分类。 |
| 数据集 | 已定义输入格式，真实数据暂未纳入版本控制 | 默认输入为 `data/raw/sp500_prices.csv`，字段为 `date,ticker,sector,close`；仓库避免提交大型原始数据。 |
| 技术路线 | 已实现 | 价格面板 → 清洗 → 收益矩阵 → 滚动窗口特征 → 标准化 → 多算法聚类 → 多指标评价 → 画像解释 → 可视化。 |
| 代码 | 已有两条路径 | `src/dynamic_market_regime.py` 是主流程；`src/market_clustering.py` 保留早期股票层面聚类脚本，便于对照。 |
| Notebook | 已有 | `notebooks/sp500_clustering_analysis.ipynb` 用于课堂说明和交互式复现实验。 |
| 前端展示 | 已有 | React/Vite 看板读取公开 CSV 和结果文件，适合现场展示图表与聚类结论。 |
| 结果目录 | 已补齐 | `results/metrics.csv`、`results/cluster_profiles.md`、`results/figures/` 等产物由脚本生成。 |
| 仍需人工补充 | 成员姓名、真实数据来源截图/下载记录 | 仓库中没有真实成员信息，也没有外部数据授权文件；答辩前需替换占位信息。 |

## 2. 项目背景与研究动机

金融市场并不是一组孤立股票的简单集合。不同阶段中，股票收益、风险、行业分化、相关性网络和市场共同运动强度会发生明显变化。例如，危机时期往往表现为高波动、高相关和网络联动增强；科技成长行情中，市场可能由少数行业主导；加息周期中，成长股和房地产等利率敏感行业可能承压。

因此，本项目希望回答四个问题：

1. 标普500市场是否存在可被聚类自动识别的结构状态？
2. 不同市场状态在收益、风险、相关网络和行业结构上有什么差异？
3. 这些状态如何在 2020–2025 年间随 COVID、加息周期、AI 行情等事件切换？
4. 聚类结果如何转化为可解释的市场画像，用于风险监控、市场复盘、用户分层和推荐系统？

## 3. 数据来源与数据范围

### 3.1 数据来源

优先数据源为标普500成分股日频收盘价或复权收盘价面板。可选来源包括课程允许的数据集、Kaggle、Yahoo Finance、NASDAQ Data Link 或学校数据库。由于仓库不应提交大型原始数据和外部授权数据，真实原始数据应本地放置在：

```text
data/raw/sp500_prices.csv
```

如果该文件不存在，`src/dynamic_market_regime.py` 会自动生成可复现模拟面板，方便课堂展示和 CI 环境复现。

### 3.2 样本规模

默认模拟样本设置如下：

| 项目 | 默认值 |
| --- | --- |
| 股票池 | 106 只标普500代表性股票代码，覆盖 11 个 GICS 行业 |
| 时间范围 | 2020-01-01 至 2025-06-30 的工作日 |
| 滚动窗口 | 60 个交易日 |
| 滚动步长 | 20 个交易日 |
| 聚类样本单位 | 每个滚动窗口形成 1 条市场快照 |
| 当前默认生成快照数 | 69 条市场快照 |

### 3.3 样本筛选规则

1. 保留字段完整的 `date,ticker,sector,close` 记录。
2. `ticker` 统一转为大写，`date` 按 `YYYY-MM-DD` 解析。
3. 只保留在全部交易日中均有价格记录的股票，避免不同窗口内股票池变化导致特征不可比。
4. 价格必须为有限正数；缺失、非数值或异常价格记录不进入有效价格序列。
5. 每个滚动窗口至少包含 `window + 1` 个价格点，才能计算 `window` 个收益率。

### 3.4 字段说明

原始数据字段：

| 字段 | 类型 | 含义 | 示例 |
| --- | --- | --- | --- |
| `date` | 日期 | 交易日期，格式 `YYYY-MM-DD` | `2023-06-30` |
| `ticker` | 字符串 | 股票代码 | `AAPL` |
| `sector` | 字符串 | 行业分类，建议使用 GICS 一级行业 | `Information Technology` |
| `close` | 数值 | 收盘价或复权收盘价 | `189.25` |

处理后市场快照字段：

| 字段 | 含义 |
| --- | --- |
| `window_id` | 窗口结束月份，用于时间轴展示 |
| `start_date`, `end_date` | 滚动窗口起止日期 |
| `mean_return`, `median_return` | 窗口内个股累计收益的均值和中位数 |
| `return_std` | 个股累计收益的横截面标准差 |
| `market_volatility` | 等权市场日收益的年化波动率 |
| `market_max_drawdown` | 窗口内等权市场净值最大回撤 |
| `return_dispersion` | 每日个股收益横截面离散度的窗口均值 |
| `volatility_dispersion` | 个股年化波动率的横截面离散度 |
| `mean_correlation`, `correlation_std` | 个股收益相关矩阵的均值和标准差 |
| `avg_degree`, `network_density`, `clustering_coefficient` | 阈值相关网络的平均度数、网络密度和聚类系数 |
| `sector_concentration` | 行业收益贡献集中度 |
| `sector_return_std`, `sector_return_spread` | 行业平均收益的标准差和极差 |
| `technology_minus_market` | 信息技术行业相对全市场平均收益 |

## 4. 数据清洗流程

1. **读取与类型转换**：读取 CSV，解析日期、股票代码、行业和价格。
2. **重复值处理**：同一 `date + ticker` 若存在重复记录，保留可解析的最后一条；答辩前若接入真实数据，应在清洗报告中说明重复来源。
3. **缺失值处理**：缺失日期、代码或价格的记录剔除；行业缺失可暂记为 `Unknown`，但最终展示建议补齐行业。
4. **异常值处理**：价格必须为正且有限；收益特征通过滚动窗口和横截面统计降低单点异常影响。早期股票层面脚本还提供 1%/99% winsorize 逻辑。
5. **股票池一致性检查**：只保留覆盖完整日期范围的股票，避免由于新股、退市或缺失交易日造成聚类样本不可比。
6. **输出结构**：清洗和特征工程结果写入 `data/processed/market_snapshots.csv`。

## 5. 特征工程方案

本项目属于市场状态聚类，因此特征重点不是单个用户或单条文本，而是滚动窗口内“市场行为强度、偏好分布、时间活跃、稳定性/多样性”的金融市场对应物。

### 5.1 行为强度特征

- `mean_return`、`median_return`：衡量窗口内市场上涨或下跌强度。
- `market_volatility`：衡量市场整体波动强度。
- `market_max_drawdown`：衡量窗口内压力和尾部风险。

### 5.2 偏好/结构分布特征

- `sector_concentration`：衡量行情是否集中在少数行业。
- `sector_return_std`、`sector_return_spread`：衡量行业分化程度。
- `technology_minus_market`：刻画科技成长主题是否明显占优。

### 5.3 时间活跃特征

- 使用 60 个交易日滚动窗口和 20 个交易日步长，把市场从静态截面转化为动态序列。
- `window_id`、`start_date`、`end_date` 支持生成状态时间轴和事件窗口分析。

### 5.4 稳定性/多样性特征

- `return_dispersion`：衡量个股收益分散程度。
- `volatility_dispersion`：衡量风险分布是否均匀。
- `mean_correlation`、`correlation_std`：衡量共同运动强度和相关结构差异。
- `avg_degree`、`network_density`、`clustering_coefficient`：把相关矩阵转化为网络结构指标，捕捉危机时期联动增强的现象。

### 5.5 标准化与降维可视化

- 聚类前对所有特征做 z-score 标准化，避免波动率、收益率、网络度数等量纲差异影响距离计算。
- 使用 PCA 二维投影生成 `results/figures/pca_cluster_scatter.svg` 和 `results/figures/pca_market_state_scatter.svg`。
- 使用确定性近邻布局作为 t-SNE 风格展示，生成 `results/figures/tsne_market_state_scatter.svg`。

## 6. 聚类算法选择

主流程比较三类无监督聚类方法：

| 算法 | 作用 | 选择原因 |
| --- | --- | --- |
| KMeans | 基准模型 | 可解释、稳定、适合课堂展示 K 值对比和 SSE 肘部法。 |
| Agglomerative Clustering | 层次聚类 | 可作为非参数形状对照，检验结果是否依赖 KMeans。 |
| Gaussian Mixture 风格模型 | 概率聚类近似 | 用软分配思想检查簇边界，当前为标准库可运行的对角协方差近似实现。 |

搜索范围默认为 `K=2..6`。最终模型不是只凭单个指标决定，而是结合指标表现、簇数量可解释性和答辩展示粒度选择。

## 7. 聚类效果评价方法

脚本会输出完整评价表到：

```text
results/metrics.csv
results/metrics/clustering_model_selection.csv
```

评价指标包括：

| 指标 | 方向 | 解释 |
| --- | --- | --- |
| Silhouette Score | 越大越好 | 衡量样本与本簇相似、与其他簇分离的程度。 |
| Davies-Bouldin Index | 越小越好 | 衡量簇内离散与簇间距离的比值。 |
| Calinski-Harabasz Index | 越大越好 | 衡量簇间离散相对簇内离散的强度。 |
| SSE/Inertia | 越小越好 | KMeans 肘部法参考，用于观察增加 K 后误差下降是否趋缓。 |

配套图表保存为：

```text
results/figures/k_metrics.svg
```

## 8. 实验流程

1. 准备或自动生成 `data/raw/sp500_prices.csv`。
2. 读取价格面板并完成字段清洗。
3. 计算个股日收益率和窗口累计收益。
4. 按 60 日窗口、20 日步长构造市场快照。
5. 提取收益、风险、横截面、相关性、网络和行业结构特征。
6. 对特征矩阵标准化。
7. 对不同算法和不同 K 值进行聚类实验。
8. 输出评价指标表、K 值对比图、PCA/t-SNE 散点图、核心特征图。
9. 自动命名市场状态并生成 `results/cluster_profiles.md`。
10. 前端或 Notebook 读取结果进行现场展示。

## 9. 结果解释方式

聚类结果不会停留在 `Cluster 0/1/2`。脚本根据每个簇的收益、波动、最大回撤、平均相关、网络密度、行业集中度和科技相对收益自动生成中文画像名称，例如：

- `Regime A：危机共振状态`
- `Regime B：高波动调整期`
- `Regime C：低波动科技牛市`
- `Regime D：低波动牛市`

画像解释文件：

```text
results/cluster_profiles.md
results/cluster_profile_summary.csv
```

图表文件：

```text
results/figures/k_metrics.svg
results/figures/pca_cluster_scatter.svg
results/figures/tsne_market_state_scatter.svg
results/figures/cluster_feature_radar.svg
results/figures/cluster_feature_bar.svg
results/figures/regime_timeline.svg
results/figures/regime_transition_graph.svg
```

## 10. 项目分工

仓库未包含真实成员信息，因此暂用占位角色。答辩前请在 `docs/team_plan.md` 中替换真实姓名和学号。

| 成员 | 主要职责 | 交付物 |
| --- | --- | --- |
| 成员A | 数据来源、数据范围、字段说明、清洗规则 | 数据说明、清洗报告、`data/processed/market_snapshots.csv` |
| 成员B | 特征工程、聚类模型、评价指标、实验复现 | `src/dynamic_market_regime.py`、`results/metrics.csv`、评价图 |
| 成员C | 可视化、画像解释、前端展示、答辩材料 | `results/cluster_profiles.md`、`results/figures/`、`docs/presentation_outline.md` |

## 11. 时间规划

| 阶段 | 时间 | 任务 | 交付物 |
| --- | --- | --- | --- |
| 数据确认 | 第1周 | 确定数据源、股票池、时间范围、字段字典 | 数据范围说明、原始数据目录结构 |
| 清洗与特征 | 第2周 | 完成清洗、滚动窗口、市场快照特征 | `data/processed/market_snapshots.csv` |
| 聚类实验 | 第3周 | 不同算法和 K 值对比，输出评价指标 | `results/metrics.csv`、`k_metrics.svg` |
| 解释与可视化 | 第4周 | 聚类命名、画像、PCA/t-SNE、雷达图/柱状图 | `cluster_profiles.md`、`results/figures/` |
| 彩排 | 答辩前3天 | 前端演示、讲稿排练、准备追问 | 可离线展示的结果文件和截图 |
| 答辩 | 现场 | 完成研究背景、方法、结果、价值和局限展示 | 最终展示 |

## 12. 现场展示方案

1. 用 README 说明研究问题和数据范围。
2. 打开 `data/processed/market_snapshots.csv` 展示市场快照字段。
3. 运行 `python src/dynamic_market_regime.py`，证明结果可复现。
4. 展示 `results/metrics.csv` 和 `results/figures/k_metrics.svg`，说明如何选择 K。
5. 展示 PCA/t-SNE 聚类散点图，说明状态可分性。
6. 展示 `cluster_feature_radar.svg` 或 `cluster_feature_bar.svg`，解释各状态核心差异。
7. 展示 `results/cluster_profiles.md`，说明可解释命名和应用价值。
8. 如果现场允许，运行 React 看板进行交互式展示；否则使用 `results/figures/` 中的 SVG 静态图。

详细逐页讲稿见 `docs/presentation_outline.md`。

## 13. 如何运行项目

### 13.1 Python 实验流程

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/dynamic_market_regime.py --start 2020-01-01 --end 2025-06-30 --window 60 --step 20
```

最小环境下也可以直接运行主脚本，因为它只依赖 Python 标准库：

```bash
python src/dynamic_market_regime.py
```

### 13.2 Notebook

```bash
jupyter notebook notebooks/sp500_clustering_analysis.ipynb
```

Notebook 应按从上到下顺序运行。若要完全复现实验结果，建议先运行主脚本生成最新 `data/processed/` 和 `results/` 文件。

### 13.3 前端展示

```bash
npm install
npm run dev
```

生产构建检查：

```bash
npm run build
```

## 14. 目录结构

```text
.
├── data/
│   ├── raw/                 # 原始数据位置；大型真实数据不提交
│   └── processed/           # 清洗后市场快照
├── docs/
│   ├── presentation_outline.md
│   └── team_plan.md
├── notebooks/
│   └── sp500_clustering_analysis.ipynb
├── results/
│   ├── metrics.csv
│   ├── cluster_profiles.md
│   ├── cluster_profile_summary.csv
│   └── figures/
├── src/
│   ├── dynamic_market_regime.py
│   ├── market_clustering.py
│   └── React 前端代码
├── requirements.txt
└── README.md
```

## 15. 局限与下一步

- 当前仓库没有提交真实大型原始数据，答辩前应补充数据下载来源、下载日期和采集说明。
- 默认模拟数据用于保证可复现，不能替代最终真实金融结论；正式答辩应尽量接入真实复权价格。
- 当前 GMM 为标准库兜底近似实现；如环境允许，可用 scikit-learn 的 `GaussianMixture`、`KMeans` 和 `AgglomerativeClustering` 替换。
- 可加入成交量、市值、估值、新闻情绪、宏观变量和社区发现算法，增强解释力。
- 成员姓名、学号和真实分工仍需人工补充。

## 13. 真实市场数据获取与运行

本仓库支持使用 Yahoo Finance / `yfinance` 下载最近五年的真实日频行情，并清洗为主流程需要的共同价格面板。

```bash
pip install -r requirements.txt
python src/fetch_market_data.py
python src/dynamic_market_regime.py \
  --prices data/raw/sp500_prices.csv \
  --start 2021-06-23 \
  --end 2026-06-23 \
  --window 60 \
  --step 10
```

说明：

- 若 `data/raw/sp500_prices.csv` 存在，主流程会直接使用真实数据文件。
- 若真实数据文件不存在，旧主流程可能会自动生成可复现模拟数据作为课堂兜底。
- 正式报告和前端展示应使用真实数据生成的 `data/processed/market_snapshots.csv`、`results/regime_assignments.csv`、`results/pca_market_coordinates.csv` 和 `results/metrics.csv`。
- 可通过 `results/data_download_summary.json` 判断真实数据是否成功：重点检查 `source` 是否为 `Yahoo Finance via yfinance`、`requested_start/requested_end` 是否为目标日期、`kept_ticker_count` 是否不少于 80、`final_trading_day_count` 和 `final_row_count` 是否为正数，以及 `failed_tickers` 和 `dropped_ticker_count` 是否可解释。
- 详细数据来源、清洗规则和限制见 `docs/data_source.md`。
