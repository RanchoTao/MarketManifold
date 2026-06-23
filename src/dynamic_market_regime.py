"""Dynamic S&P 500 market-regime mining pipeline.

This script upgrades the original stock-level clustering demo into a market-level
rolling-window study.  It intentionally depends only on the Python standard
library so the project can run in restricted classroom/CI environments.  If a
real adjusted-price panel exists at ``data/raw/sp500_prices.csv`` with columns
``date,ticker,sector,close``, the script uses it; otherwise it creates a
reproducible S&P-500-like synthetic panel that embeds COVID, tightening-cycle
and AI-led regimes for demonstration.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics as stats
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from market_config import SECTORS
except ImportError:
    from src.market_config import SECTORS

EVENTS = [
    ("COVID shock", date(2020, 2, 20), date(2020, 4, 30)),
    ("Policy/liquidity rebound", date(2020, 5, 1), date(2021, 12, 31)),
    ("Fed tightening cycle", date(2022, 1, 1), date(2022, 12, 31)),
    ("AI-led growth rally", date(2023, 1, 1), date(2024, 12, 31)),
    ("Late-cycle normalization", date(2025, 1, 1), date(2025, 6, 30)),
]
FEATURE_COLUMNS = [
    "mean_return", "median_return", "return_std", "market_volatility", "market_max_drawdown",
    "return_dispersion", "volatility_dispersion", "mean_correlation", "correlation_std",
    "avg_degree", "network_density", "clustering_coefficient", "sector_concentration",
    "sector_return_std", "sector_return_spread", "technology_minus_market",
]
PALETTE = [(37,99,235),(220,38,38),(22,163,74),(147,51,234),(234,88,12),(8,145,178),(190,18,60),(79,70,229)]

@dataclass
class Snapshot:
    window_id: str
    start: date
    end: date
    features: Dict[str, float]


def business_days(start: date, end: date) -> List[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def regime_for(d: date) -> str:
    for name, lo, hi in EVENTS:
        if lo <= d <= hi:
            return name
    return "Pre-COVID expansion"


def simulate_prices(path: Path, start: date, end: date) -> None:
    random.seed(20260607)
    days = business_days(start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    tickers = [(t, s) for s, names in SECTORS.items() for t in names]
    prices = {t: 100.0 * random.uniform(0.7, 1.4) for t, _ in tickers}
    sector_beta = {s: random.uniform(0.75, 1.25) for s in SECTORS}
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "sector", "close"])
        for d in days:
            reg = regime_for(d)
            if reg == "COVID shock":
                drift, market_sigma, corr = -0.0028, 0.034, 0.86
            elif reg == "Policy/liquidity rebound":
                drift, market_sigma, corr = 0.0010, 0.014, 0.48
            elif reg == "Fed tightening cycle":
                drift, market_sigma, corr = -0.0007, 0.020, 0.68
            elif reg == "AI-led growth rally":
                drift, market_sigma, corr = 0.0008, 0.012, 0.42
            elif reg == "Late-cycle normalization":
                drift, market_sigma, corr = 0.00025, 0.010, 0.35
            else:
                drift, market_sigma, corr = 0.00035, 0.011, 0.40
            market_shock = random.gauss(drift, market_sigma)
            for ticker, sector in tickers:
                sector_tilt = 0.0
                if reg == "AI-led growth rally" and sector in {"Information Technology", "Communication Services"}:
                    sector_tilt = 0.0012
                if reg == "Fed tightening cycle" and sector in {"Information Technology", "Consumer Discretionary", "Real Estate"}:
                    sector_tilt = -0.0008
                if reg == "COVID shock" and sector in {"Energy", "Financials", "Industrials", "Consumer Discretionary"}:
                    sector_tilt = -0.0015
                if reg == "COVID shock" and sector in {"Consumer Staples", "Health Care", "Utilities"}:
                    sector_tilt = 0.0007
                idio_sigma = market_sigma * max(0.35, 1.15 - corr) * random.uniform(0.85, 1.25)
                ret = sector_beta[sector] * market_shock + sector_tilt + random.gauss(0, idio_sigma)
                prices[ticker] *= max(0.50, 1 + ret)
                w.writerow([d.isoformat(), ticker, sector, f"{prices[ticker]:.4f}"])


def load_panel(path: Path) -> Tuple[List[date], Dict[str, str], Dict[str, List[float]]]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append((datetime.strptime(r["date"], "%Y-%m-%d").date(), r["ticker"].upper(), r.get("sector", "Unknown"), float(r["close"])))
    dates = sorted({r[0] for r in rows})
    idx = {d:i for i,d in enumerate(dates)}
    sectors = {}
    prices = defaultdict(lambda: [math.nan]*len(dates))
    for d,t,s,c in rows:
        sectors[t] = s or "Unknown"
        prices[t][idx[d]] = c
    # keep complete series only
    complete = {t:p for t,p in prices.items() if all(math.isfinite(x) and x > 0 for x in p)}
    return dates, sectors, complete


def pct_returns(series: Sequence[float]) -> List[float]:
    return [series[i]/series[i-1]-1 for i in range(1, len(series))]


def max_drawdown(values: Sequence[float]) -> float:
    peak, worst = values[0], 0.0
    for v in values:
        peak = max(peak, v)
        worst = min(worst, v/peak - 1)
    return worst


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    da = [x-ma for x in a]; db = [x-mb for x in b]
    va = sum(x*x for x in da); vb = sum(x*x for x in db)
    return sum(x*y for x,y in zip(da,db)) / math.sqrt(va*vb) if va > 0 and vb > 0 else 0.0


def safe_stdev(x: Sequence[float]) -> float:
    return stats.stdev(x) if len(x) > 1 else 0.0


def network_metrics(corr: List[List[float]], threshold: float = 0.55) -> Tuple[float,float,float]:
    n = len(corr); adj = [[False]*n for _ in range(n)]; edges = 0
    for i in range(n):
        for j in range(i+1,n):
            if corr[i][j] >= threshold:
                adj[i][j] = adj[j][i] = True; edges += 1
    degrees = [sum(row) for row in adj]
    density = 2*edges/(n*(n-1)) if n > 1 else 0.0
    coeffs=[]
    for i in range(n):
        neigh=[j for j,v in enumerate(adj[i]) if v]
        k=len(neigh)
        if k < 2:
            coeffs.append(0.0); continue
        links=sum(1 for a in range(k) for b in range(a+1,k) if adj[neigh[a]][neigh[b]])
        coeffs.append(2*links/(k*(k-1)))
    return sum(degrees)/n, density, sum(coeffs)/n


def build_snapshots(dates, sectors, prices, window=60, step=20) -> List[Snapshot]:
    tickers = sorted(prices)
    snapshots=[]
    for start_i in range(0, len(dates)-window, step):
        end_i = start_i + window
        win_dates = dates[start_i:end_i+1]
        returns = {t:pct_returns(prices[t][start_i:end_i+1]) for t in tickers}
        stock_total = {t: prices[t][end_i]/prices[t][start_i]-1 for t in tickers}
        stock_vol = {t: safe_stdev(returns[t]) * math.sqrt(252) for t in tickers}
        market_path=[]
        base=1.0
        market_rets=[]
        for k in range(window):
            r=sum(returns[t][k] for t in tickers)/len(tickers)
            market_rets.append(r); base *= (1+r); market_path.append(base)
        daily_disp=[safe_stdev([returns[t][k] for t in tickers]) for k in range(window)]
        # correlations
        pairs=[]; corrmat=[[1.0]*len(tickers) for _ in tickers]
        for i,t1 in enumerate(tickers):
            for j in range(i+1,len(tickers)):
                c=pearson(returns[t1], returns[tickers[j]])
                corrmat[i][j]=corrmat[j][i]=c; pairs.append(c)
        avg_degree,density,clustering=network_metrics(corrmat)
        sector_returns=defaultdict(list)
        for t,r in stock_total.items(): sector_returns[sectors.get(t,"Unknown")].append(r)
        sector_mean={s:sum(v)/len(v) for s,v in sector_returns.items()}
        abs_sum=sum(abs(v) for v in sector_mean.values()) or 1.0
        sector_conc=sum((abs(v)/abs_sum)**2 for v in sector_mean.values())
        tech=sector_mean.get("Information Technology", 0.0)
        features={
            "mean_return": sum(stock_total.values())/len(stock_total),
            "median_return": stats.median(stock_total.values()),
            "return_std": safe_stdev(list(stock_total.values())),
            "market_volatility": safe_stdev(market_rets)*math.sqrt(252),
            "market_max_drawdown": max_drawdown(market_path),
            "return_dispersion": sum(daily_disp)/len(daily_disp),
            "volatility_dispersion": safe_stdev(list(stock_vol.values())),
            "mean_correlation": sum(pairs)/len(pairs),
            "correlation_std": safe_stdev(pairs),
            "avg_degree": avg_degree,
            "network_density": density,
            "clustering_coefficient": clustering,
            "sector_concentration": sector_conc,
            "sector_return_std": safe_stdev(list(sector_mean.values())),
            "sector_return_spread": max(sector_mean.values())-min(sector_mean.values()),
            "technology_minus_market": tech - (sum(stock_total.values())/len(stock_total)),
        }
        snapshots.append(Snapshot(win_dates[-1].strftime("%Y-%m"), win_dates[0], win_dates[-1], features))
    return snapshots


def standardize_matrix(rows: List[Dict[str,float]], cols: List[str]) -> Tuple[List[List[float]], Dict[str,Tuple[float,float]]]:
    params={}; mat=[]
    for c in cols:
        vals=[r[c] for r in rows]; m=sum(vals)/len(vals); s=safe_stdev(vals) or 1.0; params[c]=(m,s)
    for r in rows:
        mat.append([(r[c]-params[c][0])/params[c][1] for c in cols])
    return mat, params

def sqdist(a,b): return sum((x-y)**2 for x,y in zip(a,b))
def dist(a,b): return math.sqrt(sqdist(a,b))

def kmeans(data,k,max_iter=100):
    n=len(data); cent=[data[round(i*(n-1)/(k-1))][:] for i in range(k)]; labels=[0]*n
    for _ in range(max_iter):
        old=labels[:]
        labels=[min(range(k), key=lambda c:sqdist(p,cent[c])) for p in data]
        for c in range(k):
            pts=[p for p,l in zip(data,labels) if l==c]
            if pts: cent[c]=[sum(p[j] for p in pts)/len(pts) for j in range(len(data[0]))]
        if labels==old: break
    return labels, cent

def agglomerative(data,k):
    clusters=[[i] for i in range(len(data))]
    while len(clusters)>k:
        best=(10**9,0,1)
        for i in range(len(clusters)):
            for j in range(i+1,len(clusters)):
                d=sum(dist(data[a],data[b]) for a in clusters[i] for b in clusters[j])/(len(clusters[i])*len(clusters[j]))
                if d<best[0]: best=(d,i,j)
        _,i,j=best; clusters[i]+=clusters[j]; del clusters[j]
    labels=[0]*len(data)
    for c,members in enumerate(clusters):
        for i in members: labels[i]=c
    cent=[]
    for c in range(k):
        pts=[data[i] for i,l in enumerate(labels) if l==c]
        cent.append([sum(p[j] for p in pts)/len(pts) for j in range(len(data[0]))])
    return labels, cent

def gmm_like(data,k):
    # diagonal-covariance soft refinement initialized from KMeans; suitable fallback without sklearn.
    labels,cent=kmeans(data,k)
    vars_=[[1.0]*len(data[0]) for _ in range(k)]
    weights=[1/k]*k
    for _ in range(35):
        resp=[]
        for p in data:
            probs=[]
            for c in range(k):
                expo=sum((p[j]-cent[c][j])**2/(2*max(vars_[c][j],1e-6)) for j in range(len(p)))
                det=math.prod(math.sqrt(max(v,1e-6)) for v in vars_[c])
                probs.append(weights[c]*math.exp(-expo)/(det or 1e-9))
            s=sum(probs) or 1.0; resp.append([x/s for x in probs])
        nk=[sum(r[c] for r in resp) for c in range(k)]
        for c in range(k):
            if nk[c] < 1e-6: continue
            weights[c]=nk[c]/len(data)
            cent[c]=[sum(resp[i][c]*data[i][j] for i in range(len(data)))/nk[c] for j in range(len(data[0]))]
            vars_[c]=[sum(resp[i][c]*(data[i][j]-cent[c][j])**2 for i in range(len(data)))/nk[c]+1e-4 for j in range(len(data[0]))]
    labels=[max(range(k), key=lambda c: resp[i][c]) for i in range(len(data))]
    return labels, cent

def group_indices(labels):
    g=defaultdict(list)
    for i,l in enumerate(labels): g[l].append(i)
    return g

def silhouette(data, labels):
    g=group_indices(labels); vals=[]
    if len(g)<2: return 0.0
    for i,p in enumerate(data):
        own=labels[i]; same=[j for j in g[own] if j!=i]
        a=sum(dist(p,data[j]) for j in same)/len(same) if same else 0.0
        b=min(sum(dist(p,data[j]) for j in mem)/len(mem) for lab,mem in g.items() if lab!=own)
        vals.append((b-a)/max(a,b) if max(a,b)>0 else 0.0)
    return sum(vals)/len(vals)

def dbi(data, labels, cent):
    g=group_indices(labels); scat={c:sum(dist(data[i],cent[c]) for i in mem)/len(mem) for c,mem in g.items()}
    return sum(max((scat[c]+scat[d])/(dist(cent[c],cent[d]) or 1e-9) for d in g if d!=c) for c in g)/len(g)

def ch(data, labels, cent):
    n=len(data); g=group_indices(labels); k=len(g); overall=[sum(p[j] for p in data)/n for j in range(len(data[0]))]
    between=sum(len(mem)*sqdist(cent[c],overall) for c,mem in g.items())
    within=sum(sqdist(data[i],cent[labels[i]]) for i in range(n))
    return (between/(k-1))/(within/(n-k)) if k>1 and n>k and within>0 else 0.0

def pca2(data):
    # Power iteration on covariance for first two PCs.
    m=len(data[0]); cov=[[sum(p[i]*p[j] for p in data)/(len(data)-1) for j in range(m)] for i in range(m)]
    comps=[]
    for seed in (0,1):
        v=[0.0]*m; v[seed % m]=1.0
        for _ in range(80):
            nv=[sum(cov[i][j]*v[j] for j in range(m)) for i in range(m)]
            norm=math.sqrt(sum(x*x for x in nv)) or 1.0; v=[x/norm for x in nv]
        lam=sum(v[i]*sum(cov[i][j]*v[j] for j in range(m)) for i in range(m))
        comps.append(v)
        for i in range(m):
            for j in range(m): cov[i][j]-=lam*v[i]*v[j]
    return [[sum(p[j]*comps[0][j] for j in range(m)), sum(p[j]*comps[1][j] for j in range(m))] for p in data]

def tsne_fallback(data):
    # deterministic neighbor-preserving layout seeded by PCA, then repulsive/attractive relaxation.
    xy=pca2(data)
    n=len(xy)
    for _ in range(120):
        forces=[[0.0,0.0] for _ in range(n)]
        for i in range(n):
            dists=sorted((dist(data[i],data[j]),j) for j in range(n) if j!=i)[:8]
            neigh={j for _,j in dists}
            for j in range(i+1,n):
                dx=xy[i][0]-xy[j][0]; dy=xy[i][1]-xy[j][1]; r2=dx*dx+dy*dy+1e-4
                rep=0.002/r2
                forces[i][0]+=dx*rep; forces[i][1]+=dy*rep; forces[j][0]-=dx*rep; forces[j][1]-=dy*rep
                if j in neigh:
                    forces[i][0]-=dx*0.003; forces[i][1]-=dy*0.003; forces[j][0]+=dx*0.003; forces[j][1]+=dy*0.003
        for i in range(n):
            xy[i][0]+=forces[i][0]; xy[i][1]+=forces[i][1]
    return xy

def name_regimes(snapshots, labels):
    profiles={}
    for lab, idxs in group_indices(labels).items():
        avg={c:sum(snapshots[i].features[c] for i in idxs)/len(idxs) for c in FEATURE_COLUMNS}
        r=avg['mean_return']; vol=avg['market_volatility']; corr=avg['mean_correlation']; tech=avg['technology_minus_market']; dd=avg['market_max_drawdown']
        if (vol > 0.32 and dd < -0.10) or dd < -0.14:
            name="危机共振状态"
        elif r < 0 and vol > 0.18:
            name="高波动调整期"
        elif tech > 0.03 and r > 0 and vol < 0.18:
            name="低波动科技牛市"
        elif tech > 0.03 and r > 0:
            name="科技成长驱动阶段"
        elif r > 0.10:
            name="强复苏扩张阶段"
        elif r > 0 and vol < 0.17 and corr < 0.58:
            name="低波动牛市"
        elif r > 0:
            name="复苏扩张阶段"
        else:
            name="防御震荡阶段"
        profiles[lab]=(name,avg,idxs)
    # enforce unique display names
    seen=Counter(); out={}
    letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for n,lab in enumerate(sorted(profiles)):
        name,avg,idxs=profiles[lab]; seen[name]+=1
        suffix=f" {seen[name]}" if seen[name]>1 else ""
        out[lab]=(f"Regime {letters[n]}：{name}{suffix}", avg, idxs)
    return out

# Minimal SVG drawing primitives. SVG is plain text, so generated figures are
# reviewable in pull requests and do not introduce binary artifacts.
def svg_color(rgb: Tuple[int, int, int]) -> str:
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def scale_points(points, w, h, pad=50):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    return [
        (
            pad + (x - minx) / (maxx - minx or 1) * (w - 2 * pad),
            h - pad - (y - miny) / (maxy - miny or 1) * (h - 2 * pad),
        )
        for x, y in points
    ]


def save_svg(path: Path, w: int, h: int, elements: Sequence[str], title: str) -> None:
    path.write_text(
        '\n'.join([
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{title}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<title>{title}</title>',
            *elements,
            '</svg>',
        ]),
        encoding='utf-8',
    )


def scatter_svg(points, labels, path: Path, title: str) -> None:
    w, h = 900, 620
    elements = [
        '<line x1="50" y1="570" x2="850" y2="570" stroke="#64748b" stroke-width="1"/>',
        '<line x1="50" y1="50" x2="50" y2="570" stroke="#64748b" stroke-width="1"/>',
        f'<text x="60" y="35" font-size="22" font-family="Arial" fill="#0f172a">{title}</text>',
    ]
    for (x, y), lab in zip(scale_points(points, w, h), labels):
        elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{svg_color(PALETTE[lab % len(PALETTE)])}" opacity="0.86"/>')
    save_svg(path, w, h, elements, title)


def timeline_svg(snapshots, labels, path: Path) -> None:
    w, h = 1100, 280
    y = 120
    n = len(labels)
    elements = [
        '<text x="50" y="35" font-size="22" font-family="Arial" fill="#0f172a">Market regime timeline</text>',
        '<line x1="50" y1="190" x2="1050" y2="190" stroke="#94a3b8" stroke-width="1"/>',
    ]
    for i, lab in enumerate(labels):
        x0 = 50 + i * 1000 / n
        x1 = 50 + (i + 1) * 1000 / n
        elements.append(f'<rect x="{x0:.2f}" y="{y-28}" width="{x1-x0:.2f}" height="56" fill="{svg_color(PALETTE[lab % len(PALETTE)])}"/>')
    total = (snapshots[-1].end - snapshots[0].start).days
    for name, lo, hi in EVENTS:
        x = 50 + (lo - snapshots[0].start).days / max(total, 1) * 1000
        elements.append(f'<line x1="{x:.2f}" y1="55" x2="{x:.2f}" y2="205" stroke="#1e293b" stroke-width="1.5"/>')
        elements.append(f'<text x="{x+4:.2f}" y="52" font-size="11" font-family="Arial" fill="#334155">{name}</text>')
    save_svg(path, w, h, elements, 'Market regime timeline')


def transition_svg(labels, path: Path) -> None:
    w, h = 900, 620
    labs = sorted(set(labels))
    pos = {lab: (450 + 220 * math.cos(2 * math.pi * i / len(labs)), 300 + 180 * math.sin(2 * math.pi * i / len(labs))) for i, lab in enumerate(labs)}
    counts = Counter(zip(labels, labels[1:]))
    elements = ['<text x="50" y="35" font-size="22" font-family="Arial" fill="#0f172a">Regime transition graph</text>']
    for (a, b), cnt in counts.items():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        width = min(6, 1 + cnt)
        elements.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="#475569" stroke-width="{width}" opacity="0.45"/>')
    for lab, (x, y) in pos.items():
        elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="30" fill="{svg_color(PALETTE[lab % len(PALETTE)])}"/>')
        elements.append(f'<text x="{x-4:.2f}" y="{y+5:.2f}" font-size="15" font-family="Arial" fill="white">{lab}</text>')
    save_svg(path, w, h, elements, 'Regime transition graph')


def network_svg(corr, labels, path: Path) -> None:
    w, h = 900, 720
    n = len(corr)
    pos = [(450 + 280 * math.cos(2 * math.pi * i / n), 360 + 260 * math.sin(2 * math.pi * i / n)) for i in range(n)]
    elements = ['<text x="50" y="35" font-size="22" font-family="Arial" fill="#0f172a">Latest-window correlation network</text>']
    for i in range(n):
        for j in range(i + 1, n):
            if corr[i][j] > 0.62:
                x0, y0 = pos[i]
                x1, y1 = pos[j]
                elements.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" stroke="#cbd5e1" stroke-width="0.8" opacity="0.55"/>')
    for i, (x, y) in enumerate(pos):
        elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{svg_color(PALETTE[labels[i] % len(PALETTE)])}"/>')
    save_svg(path, w, h, elements, 'Latest-window correlation network')


def bar_compare_svg(profiles, path: Path) -> None:
    w, h = 1000, 650
    cols = ['mean_return', 'market_volatility', 'mean_correlation', 'network_density', 'sector_concentration']
    labs = sorted(profiles)
    maxv = {c: max(abs(profiles[lab][1][c]) for lab in labs) or 1 for c in cols}
    elements = ['<text x="50" y="35" font-size="22" font-family="Arial" fill="#0f172a">Key period structure comparison</text>']
    for ci, col in enumerate(cols):
        x = 90 + ci * 175
        elements.append(f'<line x1="{x}" y1="560" x2="{x}" y2="80" stroke="#cbd5e1" stroke-width="1"/>')
        elements.append(f'<text x="{x-45}" y="600" font-size="11" font-family="Arial" fill="#475569">{col}</text>')
        for ri, lab in enumerate(labs):
            val = profiles[lab][1][col] / maxv[col]
            height = abs(val) * 210
            y0 = 560 - height if val >= 0 else 560
            color = svg_color(PALETTE[lab % len(PALETTE)])
            elements.append(f'<rect x="{x + ri * 18}" y="{y0:.2f}" width="14" height="{height:.2f}" fill="{color}" opacity="0.84"/>')
    save_svg(path, w, h, elements, 'Key period structure comparison')



def inertia_score(data, labels, centroids) -> float:
    """Return within-cluster sum of squared errors for elbow-method comparison."""
    return sum(sqdist(point, centroids[label]) for point, label in zip(data, labels))


def k_metrics_svg(metrics, path: Path) -> None:
    """Draw a compact K-value metric comparison chart from the model-selection table."""
    w, h = 1050, 680
    algos = sorted({row['algorithm'] for row in metrics})
    ks = sorted({int(row['k']) for row in metrics})
    # Normalize each metric to [0, 1] for a shared visual scale; DBI is inverted
    # because smaller values are better.
    raw = {
        'Silhouette ↑': [float(r['silhouette']) for r in metrics],
        'DBI ↓': [float(r['davies_bouldin']) for r in metrics],
        'CH ↑': [float(r['calinski_harabasz']) for r in metrics],
        'SSE ↓': [float(r['sse_inertia']) for r in metrics],
    }
    ranges = {m: (min(vals), max(vals)) for m, vals in raw.items()}

    def norm(metric: str, value: float) -> float:
        lo, hi = ranges[metric]
        scaled = (value - lo) / (hi - lo or 1.0)
        return 1.0 - scaled if '↓' in metric else scaled

    metric_order = ['Silhouette ↑', 'DBI ↓', 'CH ↑', 'SSE ↓']
    elements = [
        '<text x="45" y="36" font-size="22" font-family="Arial" fill="#0f172a">K-value clustering metrics comparison</text>',
        '<text x="45" y="62" font-size="12" font-family="Arial" fill="#475569">Values are normalized for display; arrows indicate preferred direction.</text>',
    ]
    chart_w, chart_h = 440, 245
    for mi, metric in enumerate(metric_order):
        ox = 55 + (mi % 2) * 500
        oy = 105 + (mi // 2) * 285
        elements.extend([
            f'<text x="{ox}" y="{oy-18}" font-size="16" font-family="Arial" fill="#0f172a">{metric}</text>',
            f'<line x1="{ox}" y1="{oy+chart_h}" x2="{ox+chart_w}" y2="{oy+chart_h}" stroke="#94a3b8"/>',
            f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+chart_h}" stroke="#94a3b8"/>',
        ])
        for k in ks:
            x = ox + (k - min(ks)) / max(1, max(ks) - min(ks)) * chart_w
            elements.append(f'<text x="{x-6:.2f}" y="{oy+chart_h+20}" font-size="10" font-family="Arial" fill="#64748b">{k}</text>')
        for ai, algo in enumerate(algos):
            pts = []
            for k in ks:
                row = next(r for r in metrics if r['algorithm'] == algo and int(r['k']) == k)
                field = {'Silhouette ↑':'silhouette','DBI ↓':'davies_bouldin','CH ↑':'calinski_harabasz','SSE ↓':'sse_inertia'}[metric]
                x = ox + (k - min(ks)) / max(1, max(ks) - min(ks)) * chart_w
                y = oy + chart_h - norm(metric, float(row[field])) * chart_h
                pts.append((x, y))
            color = svg_color(PALETTE[ai % len(PALETTE)])
            d = ' '.join(f'{x:.2f},{y:.2f}' for x, y in pts)
            elements.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2" opacity="0.88"/>')
            for x, y in pts:
                elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')
    for ai, algo in enumerate(algos):
        y = 635 + (ai // 3) * 18
        x = 55 + (ai % 3) * 250
        color = svg_color(PALETTE[ai % len(PALETTE)])
        elements.append(f'<rect x="{x}" y="{y-10}" width="12" height="12" fill="{color}"/>')
        elements.append(f'<text x="{x+18}" y="{y}" font-size="12" font-family="Arial" fill="#334155">{algo}</text>')
    save_svg(path, w, h, elements, 'K-value clustering metrics comparison')


def radar_svg(profiles, path: Path) -> None:
    """Draw a radar chart of standardized core feature averages for each regime."""
    w, h = 900, 760
    cx, cy, radius = 450, 380, 260
    cols = ['mean_return', 'market_volatility', 'mean_correlation', 'network_density', 'sector_concentration', 'technology_minus_market']
    labels_cn = ['收益', '波动', '相关', '网络', '行业集中', '科技超额']
    labs = sorted(profiles)
    mins = {c: min(profiles[lab][1][c] for lab in labs) for c in cols}
    maxs = {c: max(profiles[lab][1][c] for lab in labs) for c in cols}
    elements = ['<text x="50" y="38" font-size="22" font-family="Arial" fill="#0f172a">Cluster core-feature radar chart</text>']
    for ring in range(1, 6):
        r = radius * ring / 5
        pts = []
        for i in range(len(cols)):
            ang = -math.pi / 2 + 2 * math.pi * i / len(cols)
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        elements.append('<polygon points="{}" fill="none" stroke="#e2e8f0"/>'.format(' '.join(f'{x:.2f},{y:.2f}' for x,y in pts)))
    for i, label in enumerate(labels_cn):
        ang = -math.pi / 2 + 2 * math.pi * i / len(cols)
        x2, y2 = cx + radius * math.cos(ang), cy + radius * math.sin(ang)
        elements.append(f'<line x1="{cx}" y1="{cy}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#cbd5e1"/>')
        elements.append(f'<text x="{cx + (radius+24)*math.cos(ang):.2f}" y="{cy + (radius+24)*math.sin(ang):.2f}" font-size="12" font-family="Arial" fill="#334155" text-anchor="middle">{label}</text>')
    for li, lab in enumerate(labs):
        pts=[]
        for i, col in enumerate(cols):
            val = (profiles[lab][1][col] - mins[col]) / (maxs[col] - mins[col] or 1.0)
            r = radius * (0.15 + 0.85 * val)
            ang = -math.pi / 2 + 2 * math.pi * i / len(cols)
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        color = svg_color(PALETTE[lab % len(PALETTE)])
        elements.append('<polygon points="{}" fill="{}" opacity="0.10" stroke="{}" stroke-width="2"/>'.format(' '.join(f'{x:.2f},{y:.2f}' for x,y in pts), color, color))
        elements.append(f'<rect x="60" y="{92 + li*24}" width="12" height="12" fill="{color}"/>')
        elements.append(f'<text x="80" y="{103 + li*24}" font-size="12" font-family="Arial" fill="#334155">{profiles[lab][0]}</text>')
    save_svg(path, w, h, elements, 'Cluster core-feature radar chart')


def write_cluster_profiles(path: Path, best, profiles, snapshots) -> None:
    """Write a defendable Chinese explanation document for all discovered regimes."""
    with path.open('w', encoding='utf-8') as f:
        f.write('# 聚类市场状态画像（自动生成）\n\n')
        f.write(f"最佳模型：{best['algorithm']}，K={best['k']}。聚类对象是滚动时间窗口生成的市场快照，不是单只股票。\n\n")
        f.write('## 解释逻辑\n\n')
        f.write('- 先用标准化后的市场快照特征完成无监督聚类，再回到原始量纲解释各簇。\n')
        f.write('- 命名依据包括窗口收益、年化波动、最大回撤、横截面离散度、平均相关、网络密度、行业集中度和科技行业相对收益。\n')
        f.write('- 因为当前默认数据可由脚本模拟生成，画像应理解为课程展示版结论；接入真实复权价格后需重新运行并复核命名。\n\n')
        for lab,(name,avg,idxs) in profiles.items():
            f.write(f"## {name}\n\n")
            f.write(f"- 核心特征：平均窗口收益 {avg['mean_return']:.2%}，年化市场波动率 {avg['market_volatility']:.2%}，最大回撤 {avg['market_max_drawdown']:.2%}，平均相关 {avg['mean_correlation']:.2f}，网络密度 {avg['network_density']:.2f}，科技相对市场 {avg['technology_minus_market']:.2%}。\n")
            f.write(f"- 典型时间窗口：{', '.join(snapshots[i].window_id for i in idxs[:10])}。\n")
            if avg['market_volatility'] > 0.25 or avg['market_max_drawdown'] < -0.12:
                behavior = '市场共同下跌或快速修复时的高相关、高波动状态，分散化效果下降，风险监控优先级最高。'
            elif avg['technology_minus_market'] > 0.03:
                behavior = '科技与通信服务相对占优，市场上涨更多由成长主题和行业分化推动。'
            elif avg['mean_return'] > 0 and avg['market_volatility'] < 0.17:
                behavior = '收益为正且波动较低，适合解释为稳态扩张或低波动牛市。'
            elif avg['mean_return'] < 0:
                behavior = '收益承压且风险偏高，代表加息、估值压缩或宏观压力阶段。'
            else:
                behavior = '收益和风险处于中间状态，通常是防御震荡或状态切换过渡期。'
            f.write(f"- 典型行为/偏好：{behavior}\n")
            f.write('- 应用价值：可用于市场状态预警、组合再平衡、行业轮动复盘、投资者教育和前端可视化展示。\n\n')
        f.write('## 总体应用价值\n\n')
        f.write('- 用户分层/资产配置：不同风险承受能力的用户可对应不同市场状态下的组合策略。\n')
        f.write('- 推荐系统：当市场状态切换时，可调整相似资产召回、风险提示和行业主题推荐权重。\n')
        f.write('- 精准运营：把“危机共振”“科技驱动”“低波动扩张”等标签用于投教内容和市场复盘。\n')
        f.write('- 内容理解：为财经新闻、行情解释和仪表盘图表提供结构化语义标签。\n')

def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--prices', default='data/raw/sp500_prices.csv'); ap.add_argument('--start', default='2020-01-01'); ap.add_argument('--end', default='2025-06-30'); ap.add_argument('--window', type=int, default=60); ap.add_argument('--step', type=int, default=20); args=ap.parse_args()
    root=Path('.'); price_path=Path(args.prices)
    if not price_path.exists(): simulate_prices(price_path, datetime.strptime(args.start,'%Y-%m-%d').date(), datetime.strptime(args.end,'%Y-%m-%d').date())
    dates,sectors,prices=load_panel(price_path); snapshots=build_snapshots(dates,sectors,prices,args.window,args.step)
    rows=[{'window_id':s.window_id,'start_date':s.start.isoformat(),'end_date':s.end.isoformat(), **{c:f'{s.features[c]:.8f}' for c in FEATURE_COLUMNS}} for s in snapshots]
    write_csv(Path('data/processed/market_snapshots.csv'), rows, ['window_id','start_date','end_date']+FEATURE_COLUMNS)
    data,_=standardize_matrix([s.features for s in snapshots], FEATURE_COLUMNS)
    metrics=[]; fits={}
    algos={'kmeans':kmeans,'agglomerative':agglomerative,'gaussian_mixture':gmm_like}
    max_k=min(6, len(data)-1)
    for aname,fn in algos.items():
        for k in range(2,max_k+1):
            labels,cent=fn(data,k); sil=silhouette(data,labels); db=dbi(data,labels,cent); cal=ch(data,labels,cent)
            metrics.append({'algorithm':aname,'k':k,'silhouette':f'{sil:.6f}','davies_bouldin':f'{db:.6f}','calinski_harabasz':f'{cal:.6f}','sse_inertia':f'{inertia_score(data, labels, cent):.6f}'})
            fits[(aname,k)]=(labels,cent,sil,db,cal)
    Path('results/metrics').mkdir(parents=True, exist_ok=True)
    write_csv(Path('results/metrics/clustering_model_selection.csv'), metrics, ['algorithm','k','silhouette','davies_bouldin','calinski_harabasz','sse_inertia'])
    write_csv(Path('results/metrics.csv'), metrics, ['algorithm','k','silhouette','davies_bouldin','calinski_harabasz','sse_inertia'])
    # Select among interpretable multi-state solutions (minimum K=4) so the
    # project can distinguish crisis, tightening, low-volatility and thematic
    # leadership states instead of collapsing the whole history into only
    # crisis/non-crisis. The full K=2..6 model-selection table is still saved.
    candidates = [r for r in metrics if int(r['k']) >= min(4, max_k)]
    best=max(candidates, key=lambda r: (float(r['silhouette']), -float(r['davies_bouldin']), float(r['calinski_harabasz'])))
    labels,cent,_,_,_=fits[(best['algorithm'], int(best['k']))]
    profiles=name_regimes(snapshots,labels)
    assign=[]
    for s,l in zip(snapshots,labels): assign.append({'window_id':s.window_id,'start_date':s.start.isoformat(),'end_date':s.end.isoformat(),'regime_id':l,'regime_name':profiles[l][0]})
    write_csv(Path('results/regime_assignments.csv'), assign, ['window_id','start_date','end_date','regime_id','regime_name'])
    trans=[]
    for a,b,sa,sb in zip(labels,labels[1:],snapshots,snapshots[1:]):
        trans.append({'from_window':sa.window_id,'to_window':sb.window_id,'from_regime':profiles[a][0],'to_regime':profiles[b][0],'changed': int(a!=b)})
    write_csv(Path('results/regime_transition.csv'), trans, ['from_window','to_window','from_regime','to_regime','changed'])
    write_cluster_profiles(Path('results/cluster_profiles.md'), best, profiles, snapshots)
    write_cluster_profiles(Path('results/regime_profiles.md'), best, profiles, snapshots)
    profile_rows=[]
    for lab,(name,avg,idxs) in profiles.items():
        profile_rows.append({'regime_id':lab,'regime_name':name,'window_count':len(idxs), **{c:f'{avg[c]:.8f}' for c in FEATURE_COLUMNS}})
    write_csv(Path('results/cluster_profile_summary.csv'), profile_rows, ['regime_id','regime_name','window_count']+FEATURE_COLUMNS)
    # event analysis
    with Path('results/event_change_analysis.md').open('w',encoding='utf-8') as f:
        f.write('# 关键事件附近的结构突变检测\n\n')
        for name,lo,hi in EVENTS:
            nearby=[i for i,s in enumerate(snapshots) if abs((s.end-lo).days)<=45 or (lo<=s.end<=hi)]
            changes=sum(1 for i in nearby if i>0 and labels[i]!=labels[i-1])
            regs=sorted({profiles[labels[i]][0] for i in nearby})
            f.write(f"- **{name}**（{lo} 至 {hi}）：附近窗口 {len(nearby)} 个，状态切换 {changes} 次；识别状态：{'; '.join(regs)}。\n")
    pts = pca2(data)
    ts = tsne_fallback(data)
    fig = Path('results/figures')
    fig.mkdir(parents=True, exist_ok=True)
    scatter_svg(pts, labels, fig/'pca_market_state_scatter.svg', 'PCA market-state scatter')
    scatter_svg(pts, labels, fig/'pca_cluster_scatter.svg', 'PCA cluster scatter')
    scatter_svg(ts, labels, fig/'tsne_market_state_scatter.svg', 't-SNE-style market-state scatter')
    timeline_svg(snapshots, labels, fig/'regime_timeline.svg')
    transition_svg(labels, fig/'regime_transition_graph.svg')
    bar_compare_svg(profiles, fig/'key_period_structure_comparison.svg')
    bar_compare_svg(profiles, fig/'cluster_feature_bar.svg')
    radar_svg(profiles, fig/'cluster_feature_radar.svg')
    k_metrics_svg(metrics, fig/'k_metrics.svg')
    # latest-window stock network
    tickers=sorted(prices)[:60]; start_i=len(dates)-args.window-1; rets={t:pct_returns(prices[t][start_i:start_i+args.window+1]) for t in tickers}; corr=[[1.0]*len(tickers) for _ in tickers]
    for i,t in enumerate(tickers):
        for j in range(i+1,len(tickers)): corr[i][j]=corr[j][i]=pearson(rets[t],rets[tickers[j]])
    network_svg(corr, [labels[-1]]*len(tickers), fig/'network_structure_latest.svg')
    write_csv(Path('results/pca_market_coordinates.csv'), [{'window_id':s.window_id,'x':f'{p[0]:.6f}','y':f'{p[1]:.6f}','regime_name':profiles[l][0]} for s,p,l in zip(snapshots,pts,labels)], ['window_id','x','y','regime_name'])
    print(f"Generated {len(snapshots)} rolling market snapshots. Best model: {best['algorithm']} k={best['k']}.")

if __name__ == '__main__': main()
