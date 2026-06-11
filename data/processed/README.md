# data/processed 处理后数据目录

本目录存放清洗和特征工程后的中间结果，可提交到仓库以支持复现和展示。

## 主要文件

| 文件 | 生成方式 | 说明 |
| --- | --- | --- |
| `market_snapshots.csv` | `python src/dynamic_market_regime.py` | 滚动窗口市场快照，是动态市场状态聚类的输入。 |

## market_snapshots.csv 字段

| 字段 | 含义 |
| --- | --- |
| `window_id` | 窗口结束月份。 |
| `start_date`, `end_date` | 该市场快照对应的滚动窗口起止日期。 |
| `mean_return`, `median_return`, `return_std` | 个股窗口累计收益的均值、中位数和横截面标准差。 |
| `market_volatility`, `market_max_drawdown` | 等权市场收益的年化波动率和最大回撤。 |
| `return_dispersion`, `volatility_dispersion` | 收益和波动率的横截面分散程度。 |
| `mean_correlation`, `correlation_std` | 个股收益相关矩阵的均值和标准差。 |
| `avg_degree`, `network_density`, `clustering_coefficient` | 相关网络结构特征。 |
| `sector_concentration`, `sector_return_std`, `sector_return_spread`, `technology_minus_market` | 行业结构和科技行业相对表现特征。 |

## 清洗规则摘要

- 删除无法解析日期、代码或价格的记录。
- 价格必须为有限正数。
- 股票代码统一大写。
- 保留完整覆盖日期范围的股票，以保证窗口间可比。
- 大型原始数据不进入本目录；本目录只保存可复现实验所需的处理后特征。
