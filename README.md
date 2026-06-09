# 基于动态网络与聚类分析的标普500市场结构演化研究

本仓库已从“股票特征 + KMeans 聚类 Demo”重构为一个完整的数据挖掘研究项目：研究标普500市场结构如何形成、如何随时间演化，以及重大事件附近是否出现结构突变。聚类对象不再是单只股票，而是按滚动窗口生成的**市场快照（market snapshots）**。

## 1. 研究问题

1. **市场是否存在不同结构状态（Market Regime）？**  
   通过滚动窗口提取市场级收益、风险、横截面、相关性、网络与行业结构特征，并用无监督聚类发现状态。
2. **市场结构如何随时间变化？**  
   将每个窗口映射到一个状态，生成 2020 至 2025 年的状态时间轴与状态转移表。
3. **重大事件是否导致市场结构突变？**  
   自动检查 COVID 冲击、美联储加息周期、AI 行情等事件附近的状态切换。
4. **能否利用聚类方法自动发现市场状态？**  
   比较 KMeans、Agglomerative Clustering、Gaussian Mixture 三类方法，并用 Silhouette、Davies-Bouldin、Calinski-Harabasz 指标选择模型。

## 2. 方法依据与成熟做法

项目方法参考了金融网络与市场状态识别领域的成熟实践：

- 金融资产收益的动态相关网络常通过**滚动窗口相关矩阵**刻画市场层面的联动关系，相关结构在危机期间通常会增强。参考：Dynamic correlation network analysis of financial asset returns with network clustering, PLOS ONE / PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC6214240/
- 相关网络、MST/阈值网络和网络聚类可用于描述金融市场拓扑与行业结构，危机阶段往往表现为更强的共同运动与更高的网络连接度。参考：Dynamics of Market Correlations: Taxonomy and Portfolio Analysis: https://arxiv.org/abs/1011.3225
- 动态金融网络研究强调在连续时间窗口中跟踪拓扑指标、社区结构和市场状态变化。参考：Modular Dynamics of Financial Market Networks: https://arxiv.org/abs/1501.05040
- 多因子金融网络聚类说明行业、国家/风格等多重因素会共同塑造资产间相关结构。参考：Dynamic Multi-Factor Clustering of Financial Networks: https://arxiv.org/abs/1505.01550

## 3. 数据层设计

默认流程优先读取：

```text
data/raw/sp500_prices.csv
```

字段要求：

```text
date,ticker,sector,close
```

如果没有真实价格面板，脚本会自动生成一个可复现实验面板 `data/raw/sp500_prices.csv`，覆盖 2020-01-01 至 2025-06-30 的工作日，并内置以下市场事件：

- COVID shock：2020-02-20 至 2020-04-30
- Policy/liquidity rebound：2020-05-01 至 2021-12-31
- Fed tightening cycle：2022-01-01 至 2022-12-31
- AI-led growth rally：2023-01-01 至 2024-12-31
- Late-cycle normalization：2025-01-01 至 2025-06-30

> 说明：当前仓库没有外部 API 凭据，且执行环境网络包安装受限。因此代码采用“真实数据优先、可复现模拟数据兜底”的工程策略，保证 Notebook、脚本和结果文件均可运行复现。替换为真实数据时，只需按字段格式放入 `data/raw/sp500_prices.csv` 后重新运行脚本。

## 4. 滚动窗口市场快照

- 时间范围：默认 2020-01-01 至 2025-06-30；可通过命令行参数调整。
- 窗口长度：60 个交易日。
- 步长：20 个交易日。
- 样本单位：每个滚动窗口是一条市场状态样本。
- 输出：`data/processed/market_snapshots.csv`。

运行命令：

```bash
python src/dynamic_market_regime.py --start 2020-01-01 --end 2025-06-30 --window 60 --step 20
```

## 5. 市场状态特征工程

每个窗口提取以下市场级特征：

| 维度 | 特征 |
| --- | --- |
| 收益率 | 平均窗口收益、中位窗口收益、收益率标准差 |
| 风险 | 年化市场波动率、最大回撤 |
| 横截面 | 平均日度收益离散度、个股波动率离散度 |
| 相关性结构 | 平均相关系数、相关系数标准差 |
| 网络结构 | 平均度数、网络密度、聚类系数 |
| 行业结构 | 行业集中度、行业收益标准差、行业收益差、科技相对市场收益 |

这些特征将市场看成一个随时间变化的系统，而不是把股票静态分类。

## 6. 市场状态发现

脚本比较三类聚类方法：

- `kmeans`
- `agglomerative`
- `gaussian_mixture`（标准库环境下实现为对角协方差 GMM 风格 EM 近似）

评价指标：

- Silhouette Score：越大越好。
- Davies-Bouldin Index：越小越好。
- Calinski-Harabasz Index：越大越好。

模型选择表保存到：

```text
results/metrics/clustering_model_selection.csv
```

## 7. 市场状态解释

脚本不会只输出 Cluster 编号，而是根据收益、风险、相关性、网络与行业特征自动命名状态，例如：

- Regime A：危机共振状态
- Regime B：低波动牛市
- Regime C：高波动调整期
- Regime D：科技成长驱动阶段

完整解释保存到：

```text
results/regime_profiles.md
```

## 8. 动态演化分析

核心输出：

- `results/regime_assignments.csv`：每个窗口所属市场状态。
- `results/regime_transition.csv`：相邻窗口状态转移。
- `results/event_change_analysis.md`：COVID、加息周期、AI 行情等事件附近的结构变化检查。
- `results/figures/regime_timeline.svg`：市场状态时间轴。
- `results/figures/regime_transition_graph.svg`：状态转移图。

## 9. 可视化输出

所有新增可视化统一保存到 `results/figures/`，并使用可审阅的纯文本 SVG，避免在 PR 中继续提交新的二进制图片：

1. `pca_market_state_scatter.svg`：PCA 市场状态散点图。
2. `tsne_market_state_scatter.svg`：t-SNE 风格市场状态散点图（标准库可复现近邻布局）。
3. `regime_timeline.svg`：市场状态时间轴。
4. `regime_transition_graph.svg`：状态转移图。
5. `network_structure_latest.svg`：最新窗口相关网络结构图。
6. `key_period_structure_comparison.svg`：关键状态结构对比图。

## 10. Notebook 与现场展示

- Notebook：`notebooks/sp500_clustering_analysis.ipynb`，可直接运行并调用主脚本生成结果。
- 答辩提纲：`docs/presentation_outline.md`，不少于 10 页，覆盖研究背景、问题、数据、特征、网络、聚类、指标、状态发现、动态演化、应用价值、局限与未来工作。
- 前端看板仍保留原股票特征 Demo，可作为“原始项目对比/资产层辅助视图”；本次研究主线以 `src/dynamic_market_regime.py` 和 `results/` 产物为准。

## 11. 目录结构

```text
src/dynamic_market_regime.py          # 动态市场状态挖掘主脚本
data/raw/sp500_prices.csv             # 真实数据优先；缺失时自动生成的价格面板
data/processed/market_snapshots.csv   # 60日窗口、20日步长的市场快照特征
results/metrics/                      # 聚类模型选择指标
results/regime_profiles.md            # 自动解释后的状态画像
results/regime_transition.csv         # 状态转移明细
results/figures/                      # PCA/t-SNE/时间轴/转移图/网络图/对比图
docs/presentation_outline.md          # 现场答辩提纲
notebooks/sp500_clustering_analysis.ipynb
```

## 12. 复现步骤

```bash
# 1. 生成滚动市场快照、聚类结果、状态解释和图表
python src/dynamic_market_regime.py

# 2. 可选：运行旧版前端资产层辅助看板
npm run build
```

## 13. 当前研究结论（基于仓库可复现实验）

- 标普500市场可以被表达为少数几个动态结构状态，而不是固定不变的股票类别。
- 高压力阶段表现为更高波动率、更深回撤、更高平均相关和更高网络密度，意味着分散化收益下降。
- 低波动牛市或科技成长阶段通常具有较低网络同步性，并伴随行业收益分化。
- COVID、加息周期、AI 行情附近均可通过状态切换或状态持续性变化体现结构演化。

## 14. 下一步优化

- 接入真实 S&P 500 成分股日度复权价格和 GICS 行业分类。
- 增加 MST、PMFG、社区发现、模块度、中心性等网络指标。
- 使用真实 t-SNE/UMAP、HDBSCAN、Hidden Markov Model 和变点检测算法。
- 对事件窗口进行统计检验，例如置换检验或 bootstrap 置信区间。
- 将前端改造为市场状态时间轴交互展示，而非仅展示资产层聚类。
