#!/usr/bin/env python3
"""
BTC 四年周期理论验证 - 数据分析与图表生成
使用币安 BTCUSDT 真实数据
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor'] = '#161b22'
plt.rcParams['text.color'] = '#e6edf3'
plt.rcParams['axes.labelcolor'] = '#e6edf3'
plt.rcParams['xtick.color'] = '#8b949e'
plt.rcParams['ytick.color'] = '#8b949e'

OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ 关键历史事件 ============
HALVINGS = {
    "第一次减半": datetime(2012, 11, 28),
    "第二次减半": datetime(2016, 7, 9),
    "第三次减半": datetime(2020, 5, 11),
    "第四次减半": datetime(2024, 4, 19),
}

# 基于公开历史记录的关键价格点（币安数据从2017-08开始，更早的用已知数据标注）
CYCLE_PEAKS = {
    "周期1峰值 (2013-12)": (datetime(2013, 12, 4), 1150),
    "周期2峰值 (2017-12)": (datetime(2017, 12, 17), 19783),
    "周期3峰值 (2021-11)": (datetime(2021, 11, 10), 69000),
    "周期4峰值 (2024-10)": (datetime(2024, 10, 29), 73750),  # 将从数据中校正
}

CYCLE_TROUGHS = {
    "周期1底部 (2015-01)": (datetime(2015, 1, 14), 178),
    "周期2底部 (2018-12)": (datetime(2018, 12, 15), 3236),
    "周期3底部 (2022-11)": (datetime(2022, 11, 21), 15476),
}


def load_data():
    """加载所有时间级别的K线数据"""
    data = {}
    for interval in ["1d", "8h", "1w", "1M"]:
        filepath = f"data/btcusdt_{interval}.csv"
        df = pd.read_csv(filepath, index_col="date", parse_dates=True)
        data[interval] = df
        print(f"加载 {interval}: {len(df)} 条, {df.index[0].date()} ~ {df.index[-1].date()}")
    return data


def find_actual_peak(df):
    """从真实数据中找到最高价"""
    idx = df['high'].idxmax()
    return idx, df.loc[idx, 'high']


def chart1_price_overview_with_halvings(daily):
    """图表1: BTC 价格走势全景图 + 减半事件 + 周期标注"""
    fig, ax = plt.subplots(figsize=(20, 10))

    # 价格曲线（对数尺度）
    ax.semilogy(daily.index, daily['close'], color='#f0883e', linewidth=1.2, alpha=0.9, label='BTCUSDT 收盘价')

    # 标注减半事件
    colors_halving = ['#ff7b72', '#79c0ff', '#7ee787', '#d2a8ff']
    for i, (name, date) in enumerate(HALVINGS.items()):
        if date >= daily.index[0]:
            ax.axvline(x=date, color=colors_halving[i], linestyle='--', alpha=0.7, linewidth=1.5)
            price_at_halving = daily.loc[daily.index.asof(date), 'close'] if date >= daily.index[0] else None
            if price_at_halving:
                ax.annotate(f'{name}\n${price_at_halving:,.0f}',
                           xy=(date, price_at_halving),
                           xytext=(15, 40), textcoords='offset points',
                           fontsize=9, color=colors_halving[i], fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color=colors_halving[i], lw=1.5),
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor=colors_halving[i], alpha=0.8))

    # 标注周期峰值（仅在数据范围内的）
    for name, (date, price) in CYCLE_PEAKS.items():
        if date >= daily.index[0]:
            nearest = daily.index.asof(date)
            actual_price = daily.loc[nearest, 'high']
            ax.plot(nearest, actual_price, 'v', color='#ff4444', markersize=12, zorder=5)
            ax.annotate(f'峰值\n${actual_price:,.0f}',
                       xy=(nearest, actual_price),
                       xytext=(0, 25), textcoords='offset points',
                       fontsize=8, color='#ff4444', ha='center', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='#21262d', edgecolor='#ff4444', alpha=0.8))

    # 标注周期底部（仅在数据范围内的）
    for name, (date, price) in CYCLE_TROUGHS.items():
        if date >= daily.index[0]:
            nearest = daily.index.asof(date)
            actual_price = daily.loc[nearest, 'low']
            ax.plot(nearest, actual_price, '^', color='#7ee787', markersize=12, zorder=5)
            ax.annotate(f'底部\n${actual_price:,.0f}',
                       xy=(nearest, actual_price),
                       xytext=(0, -35), textcoords='offset points',
                       fontsize=8, color='#7ee787', ha='center', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='#21262d', edgecolor='#7ee787', alpha=0.8))

    # 周期区间着色
    cycle_ranges = [
        (datetime(2016, 7, 9), datetime(2020, 5, 11), '第三周期 (2016减半→2020减半)', '#79c0ff'),
        (datetime(2020, 5, 11), datetime(2024, 4, 19), '第四周期 (2020减半→2024减半)', '#7ee787'),
        (datetime(2024, 4, 19), datetime(2028, 4, 1), '第五周期 (2024减半→~2028)', '#d2a8ff'),
    ]
    for start, end, label, color in cycle_ranges:
        actual_end = min(end, daily.index[-1] + timedelta(days=365))
        if start >= daily.index[0]:
            ax.axvspan(start, actual_end, alpha=0.06, color=color)

    ax.set_title('BTC/USDT 价格走势全景图 — 减半周期标注 (2017-2026)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('日期', fontsize=13)
    ax.set_ylabel('价格 (USDT, 对数尺度)', fontsize=13)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.legend(loc='upper left', fontsize=11, facecolor='#21262d', edgecolor='#30363d')
    ax.grid(True, alpha=0.15, color='#30363d')
    ax.set_xlim(daily.index[0], daily.index[-1] + timedelta(days=30))

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/01_price_overview_halvings.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ 图表1已生成: {path}")
    return path


def chart2_cycle_comparison_from_halving(daily):
    """图表2: 各周期减半后价格走势对比（对齐减半日为Day 0）"""
    fig, ax = plt.subplots(figsize=(18, 10))

    halving_list = list(HALVINGS.items())
    colors = ['#ff7b72', '#79c0ff', '#7ee787', '#d2a8ff']

    for i, (name, halving_date) in enumerate(halving_list):
        if halving_date < daily.index[0]:
            continue

        # 从减半日起最多取1460天(4年)的数据
        mask = (daily.index >= halving_date) & (daily.index < halving_date + timedelta(days=1460))
        cycle_data = daily.loc[mask].copy()

        if len(cycle_data) == 0:
            continue

        # 计算相对天数和归一化价格
        days_since = (cycle_data.index - halving_date).days
        base_price = cycle_data['close'].iloc[0]
        normalized = cycle_data['close'] / base_price

        ax.plot(days_since, normalized, color=colors[i], linewidth=2, label=f'{name} (基准: ${base_price:,.0f})', alpha=0.9)

        # 标注峰值
        peak_idx = normalized.idxmax()
        peak_day = (peak_idx - halving_date).days
        peak_val = normalized.max()
        peak_price = cycle_data.loc[peak_idx, 'close']
        ax.plot(peak_day, peak_val, 'o', color=colors[i], markersize=10, zorder=5)
        ax.annotate(f'峰值: {peak_val:.1f}x\n(${peak_price:,.0f})\n第{peak_day}天',
                   xy=(peak_day, peak_val),
                   xytext=(15, 10), textcoords='offset points',
                   fontsize=9, color=colors[i], fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor=colors[i], alpha=0.8))

    ax.set_yscale('log')
    ax.set_title('各周期减半后价格走势对比 (以减半日为 Day 0，价格归一化)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('减半后天数', fontsize=13)
    ax.set_ylabel('价格倍数 (对数尺度)', fontsize=13)
    ax.axhline(y=1, color='#484f58', linestyle='--', alpha=0.5, label='基准线 (1x)')
    ax.legend(loc='upper left', fontsize=11, facecolor='#21262d', edgecolor='#30363d')
    ax.grid(True, alpha=0.15, color='#30363d')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.1f}x'))

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/02_cycle_comparison_from_halving.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ 图表2已生成: {path}")
    return path


def chart3_drawdown_analysis(daily):
    """图表3: 历史回撤分析"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), height_ratios=[1.5, 1])

    # 计算历史最高价和回撤
    daily['ath'] = daily['high'].cummax()
    daily['drawdown'] = (daily['close'] - daily['ath']) / daily['ath'] * 100

    # 上半图: 价格 + ATH
    ax1.semilogy(daily.index, daily['close'], color='#f0883e', linewidth=1, label='收盘价', alpha=0.9)
    ax1.semilogy(daily.index, daily['ath'], color='#484f58', linewidth=0.8, linestyle='--', label='历史最高价 (ATH)', alpha=0.6)
    ax1.fill_between(daily.index, daily['close'], daily['ath'], alpha=0.1, color='#ff4444')

    # 标注减半
    for name, date in HALVINGS.items():
        if date >= daily.index[0]:
            ax1.axvline(x=date, color='#d2a8ff', linestyle=':', alpha=0.5)

    ax1.set_title('BTC/USDT 历史回撤分析', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel('价格 (USDT)', fontsize=12)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.legend(loc='upper left', fontsize=10, facecolor='#21262d', edgecolor='#30363d')
    ax1.grid(True, alpha=0.15, color='#30363d')

    # 下半图: 回撤百分比
    ax2.fill_between(daily.index, daily['drawdown'], 0, color='#ff4444', alpha=0.4)
    ax2.plot(daily.index, daily['drawdown'], color='#ff7b72', linewidth=0.8)

    # 标注主要回撤
    # 找出局部最大回撤
    major_drawdowns = [
        ("2018熊市", datetime(2018, 12, 15)),
        ("2020.3暴跌", datetime(2020, 3, 13)),
        ("2022熊市", datetime(2022, 11, 21)),
    ]
    for label, date in major_drawdowns:
        if date >= daily.index[0]:
            nearest = daily.index.asof(date)
            dd = daily.loc[nearest, 'drawdown']
            ax2.annotate(f'{label}\n{dd:.1f}%',
                        xy=(nearest, dd),
                        xytext=(20, -20), textcoords='offset points',
                        fontsize=9, color='#ff7b72', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#ff7b72'),
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#21262d', edgecolor='#ff7b72', alpha=0.8))

    # 水平参考线
    for level, label in [(-20, '-20%'), (-50, '-50%'), (-80, '-80%')]:
        ax2.axhline(y=level, color='#484f58', linestyle='--', alpha=0.3)
        ax2.text(daily.index[5], level + 2, label, fontsize=8, color='#8b949e')

    ax2.set_ylabel('从ATH回撤 (%)', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylim(-100, 5)
    ax2.grid(True, alpha=0.15, color='#30363d')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/03_drawdown_analysis.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ 图表3已生成: {path}")
    return path


def chart4_monthly_returns_heatmap(monthly):
    """图表4: 月度收益率热力图"""
    monthly_returns = monthly['close'].pct_change() * 100

    # 创建年-月矩阵
    years = sorted(monthly.index.year.unique())
    months = range(1, 13)
    month_names = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']

    matrix = np.full((len(years), 12), np.nan)
    for i, year in enumerate(years):
        for j, month in enumerate(months):
            mask = (monthly.index.year == year) & (monthly.index.month == month)
            if mask.any():
                val = monthly_returns.loc[mask]
                if not val.empty and not np.isnan(val.iloc[0]):
                    matrix[i, j] = val.iloc[0]

    fig, ax = plt.subplots(figsize=(16, 8))

    # 自定义颜色映射: 红(负) -> 黑(零) -> 绿(正)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#ff4444', '#21262d', '#7ee787']
    cmap = LinearSegmentedColormap.from_list('btc', colors_list, N=256)

    vmax = np.nanmax(np.abs(matrix))
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)

    # 标注数值
    for i in range(len(years)):
        for j in range(12):
            val = matrix[i, j]
            if not np.isnan(val):
                color = '#ffffff' if abs(val) > vmax * 0.3 else '#8b949e'
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=7, color=color, fontweight='bold')

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names, fontsize=10)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years, fontsize=10)

    # 标注减半年份
    halving_years = [2020, 2024]
    for hy in halving_years:
        if hy in years:
            idx = years.index(hy)
            ax.get_yticklabels()[idx].set_color('#d2a8ff')
            ax.get_yticklabels()[idx].set_fontweight('bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='月收益率 (%)')
    cbar.ax.yaxis.label.set_color('#e6edf3')
    cbar.ax.tick_params(colors='#8b949e')

    ax.set_title('BTC/USDT 月度收益率热力图 (紫色标注=减半年份)', fontsize=16, fontweight='bold', pad=15)
    ax.grid(False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/04_monthly_returns_heatmap.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ 图表4已生成: {path}")
    return path


def chart5_volume_analysis(daily):
    """图表5: 成交量与价格关系分析"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), height_ratios=[1.5, 1])

    # 上图: 价格
    ax1.semilogy(daily.index, daily['close'], color='#f0883e', linewidth=1, alpha=0.9)

    for name, date in HALVINGS.items():
        if date >= daily.index[0]:
            ax1.axvline(x=date, color='#d2a8ff', linestyle='--', alpha=0.5, linewidth=1)

    ax1.set_title('BTC/USDT 成交量与价格关系', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel('价格 (USDT)', fontsize=12)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.grid(True, alpha=0.15, color='#30363d')

    # 下图: 成交量（USDT计价）
    # 用颜色区分涨跌
    colors_vol = ['#7ee787' if c >= o else '#ff4444' for c, o in zip(daily['close'], daily['open'])]
    ax2.bar(daily.index, daily['quote_volume'] / 1e9, color=colors_vol, alpha=0.6, width=1)

    # 30日移动平均成交量
    vol_ma30 = daily['quote_volume'].rolling(30).mean() / 1e9
    ax2.plot(daily.index, vol_ma30, color='#79c0ff', linewidth=1.5, label='30日均成交量')

    for name, date in HALVINGS.items():
        if date >= daily.index[0]:
            ax2.axvline(x=date, color='#d2a8ff', linestyle='--', alpha=0.5, linewidth=1)

    ax2.set_ylabel('成交量 (十亿 USDT)', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10, facecolor='#21262d', edgecolor='#30363d')
    ax2.grid(True, alpha=0.15, color='#30363d')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/05_volume_analysis.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ 图表5已生成: {path}")
    return path


def chart6_volatility_cycle(daily):
    """图表6: 波动率周期分析"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), height_ratios=[1, 1])

    # 计算各种波动率指标
    daily['returns'] = daily['close'].pct_change()
    daily['vol_30d'] = daily['returns'].rolling(30).std() * np.sqrt(365) * 100  # 年化
    daily['vol_90d'] = daily['returns'].rolling(90).std() * np.sqrt(365) * 100
    daily['vol_365d'] = daily['returns'].rolling(365).std() * np.sqrt(365) * 100

    # 上图: 30日和90日年化波动率
    ax1.plot(daily.index, daily['vol_30d'], color='#ff7b72', linewidth=1, alpha=0.7, label='30日年化波动率')
    ax1.plot(daily.index, daily['vol_90d'], color='#79c0ff', linewidth=1.5, alpha=0.9, label='90日年化波动率')
    ax1.plot(daily.index, daily['vol_365d'], color='#7ee787', linewidth=2, alpha=0.9, label='365日年化波动率')

    for name, date in HALVINGS.items():
        if date >= daily.index[0]:
            ax1.axvline(x=date, color='#d2a8ff', linestyle='--', alpha=0.5)

    ax1.set_title('BTC/USDT 波动率周期分析', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel('年化波动率 (%)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10, facecolor='#21262d', edgecolor='#30363d')
    ax1.grid(True, alpha=0.15, color='#30363d')

    # 下图: 按年统计波动率箱线图
    daily['year'] = daily.index.year
    years = sorted(daily['year'].unique())
    vol_by_year = [daily.loc[daily['year'] == y, 'returns'].dropna().values * 100 for y in years]

    bp = ax2.boxplot(vol_by_year, patch_artist=True, labels=years,
                     medianprops=dict(color='#f0883e', linewidth=2),
                     whiskerprops=dict(color='#8b949e'),
                     capprops=dict(color='#8b949e'),
                     flierprops=dict(marker='o', markerfacecolor='#ff4444', markersize=3, alpha=0.5))

    # 着色: 减半年份用紫色，其他用蓝色
    halving_years_set = {2020, 2024}
    for i, (box, year) in enumerate(zip(bp['boxes'], years)):
        if year in halving_years_set:
            box.set(facecolor='#d2a8ff', alpha=0.4)
        else:
            box.set(facecolor='#79c0ff', alpha=0.3)

    ax2.set_ylabel('日收益率 (%)', fontsize=12)
    ax2.set_xlabel('年份 (紫色=减半年份)', fontsize=12)
    ax2.grid(True, alpha=0.15, color='#30363d')
    ax2.axhline(y=0, color='#484f58', linestyle='-', alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=45)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/06_volatility_cycle.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ 图表6已生成: {path}")
    return path


def chart7_8h_kline_analysis(h8):
    """图表7: 8小时K线 - 日内周期分析"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    h8['hour'] = h8.index.hour
    h8['returns'] = h8['close'].pct_change()
    h8['range_pct'] = (h8['high'] - h8['low']) / h8['open'] * 100

    # 1. 各时段平均收益率
    ax = axes[0, 0]
    hourly_returns = h8.groupby('hour')['returns'].mean() * 100
    colors_bar = ['#7ee787' if v > 0 else '#ff4444' for v in hourly_returns.values]
    bars = ax.bar(hourly_returns.index.astype(str) + ':00', hourly_returns.values, color=colors_bar, alpha=0.8)
    ax.set_title('各8小时时段平均收益率', fontsize=13, fontweight='bold')
    ax.set_ylabel('平均收益率 (%)')
    ax.axhline(y=0, color='#484f58', linestyle='-', alpha=0.5)
    ax.grid(True, alpha=0.15, color='#30363d')

    # 2. 各时段波动幅度
    ax = axes[0, 1]
    hourly_range = h8.groupby('hour')['range_pct'].mean()
    ax.bar(hourly_range.index.astype(str) + ':00', hourly_range.values, color='#79c0ff', alpha=0.8)
    ax.set_title('各8小时时段平均波动幅度', fontsize=13, fontweight='bold')
    ax.set_ylabel('(高-低)/开盘 (%)')
    ax.grid(True, alpha=0.15, color='#30363d')

    # 3. 各时段成交量占比
    ax = axes[1, 0]
    h8['date_only'] = h8.index.date
    daily_vol = h8.groupby('date_only')['quote_volume'].sum()
    h8_merged = h8.copy()
    h8_merged['daily_vol'] = h8_merged['date_only'].map(daily_vol)
    h8_merged['vol_pct'] = h8_merged['quote_volume'] / h8_merged['daily_vol'] * 100
    hourly_vol = h8_merged.groupby('hour')['vol_pct'].mean()
    ax.bar(hourly_vol.index.astype(str) + ':00', hourly_vol.values, color='#f0883e', alpha=0.8)
    ax.set_title('各8小时时段成交量占比', fontsize=13, fontweight='bold')
    ax.set_ylabel('日成交量占比 (%)')
    ax.grid(True, alpha=0.15, color='#30363d')

    # 4. 时段收益率按年份演变
    ax = axes[1, 1]
    h8['year'] = h8.index.year
    for year in sorted(h8['year'].unique())[-4:]:
        yearly = h8[h8['year'] == year].groupby('hour')['returns'].mean() * 100
        ax.plot(yearly.index.astype(str) + ':00', yearly.values, marker='o', linewidth=2, label=str(year), alpha=0.8)
    ax.set_title('近4年各时段收益率对比', fontsize=13, fontweight='bold')
    ax.set_ylabel('平均收益率 (%)')
    ax.legend(facecolor='#21262d', edgecolor='#30363d')
    ax.axhline(y=0, color='#484f58', linestyle='-', alpha=0.5)
    ax.grid(True, alpha=0.15, color='#30363d')

    fig.suptitle('BTC/USDT 8小时K线 日内周期分析', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/07_8h_kline_analysis.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ 图表7已生成: {path}")
    return path


def chart8_weekly_analysis(weekly):
    """图表8: 周K线分析 - 周度趋势"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))

    # 周收益率
    weekly['returns'] = weekly['close'].pct_change() * 100

    # 上图: 周收益率柱状图
    colors_weekly = ['#7ee787' if v > 0 else '#ff4444' for v in weekly['returns'].values]
    ax1.bar(weekly.index, weekly['returns'], color=colors_weekly, alpha=0.6, width=5)

    # 添加12周均线
    ma12 = weekly['returns'].rolling(12).mean()
    ax1.plot(weekly.index, ma12, color='#79c0ff', linewidth=2, label='12周移动平均')

    for name, date in HALVINGS.items():
        if date >= weekly.index[0]:
            ax1.axvline(x=date, color='#d2a8ff', linestyle='--', alpha=0.5)

    ax1.set_title('BTC/USDT 周收益率与趋势', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel('周收益率 (%)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10, facecolor='#21262d', edgecolor='#30363d')
    ax1.grid(True, alpha=0.15, color='#30363d')
    ax1.axhline(y=0, color='#484f58', linestyle='-', alpha=0.3)

    # 下图: 连续涨/跌周统计
    weekly['direction'] = (weekly['returns'] > 0).astype(int)
    # 计算连续涨/跌周数
    streaks = []
    current_streak = 0
    current_dir = None
    for idx, row in weekly.iterrows():
        if np.isnan(row['returns']):
            continue
        d = 1 if row['returns'] > 0 else -1
        if d == current_dir:
            current_streak += d
        else:
            current_dir = d
            current_streak = d
        streaks.append((idx, current_streak))

    streak_df = pd.DataFrame(streaks, columns=['date', 'streak'])
    streak_df.set_index('date', inplace=True)

    colors_streak = ['#7ee787' if v > 0 else '#ff4444' for v in streak_df['streak']]
    ax2.bar(streak_df.index, streak_df['streak'], color=colors_streak, alpha=0.6, width=5)

    for name, date in HALVINGS.items():
        if date >= weekly.index[0]:
            ax2.axvline(x=date, color='#d2a8ff', linestyle='--', alpha=0.5)

    ax2.set_ylabel('连续涨/跌周数', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.grid(True, alpha=0.15, color='#30363d')
    ax2.axhline(y=0, color='#484f58', linestyle='-', alpha=0.3)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/08_weekly_analysis.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ 图表8已生成: {path}")
    return path


def chart9_moving_averages(daily):
    """图表9: 关键均线系统与周期"""
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.semilogy(daily.index, daily['close'], color='#e6edf3', linewidth=0.8, alpha=0.6, label='收盘价')

    # 各种均线
    mas = {
        'MA50': (50, '#ff7b72'),
        'MA100': (100, '#f0883e'),
        'MA200': (200, '#79c0ff'),
        'MA365': (365, '#7ee787'),
        'MA730 (2年)': (730, '#d2a8ff'),
    }

    for name, (period, color) in mas.items():
        ma = daily['close'].rolling(period).mean()
        ax.semilogy(daily.index, ma, color=color, linewidth=1.8, label=name, alpha=0.85)

    for hname, date in HALVINGS.items():
        if date >= daily.index[0]:
            ax.axvline(x=date, color='#ffa657', linestyle=':', alpha=0.4, linewidth=1)

    ax.set_title('BTC/USDT 关键均线系统与周期分析', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('价格 (USDT, 对数尺度)', fontsize=12)
    ax.set_xlabel('日期', fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.legend(loc='upper left', fontsize=10, facecolor='#21262d', edgecolor='#30363d', ncol=2)
    ax.grid(True, alpha=0.15, color='#30363d')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/09_moving_averages.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ 图表9已生成: {path}")
    return path


def chart10_prediction_model(daily):
    """图表10: 基于历史周期的价格预测模型"""
    fig, ax = plt.subplots(figsize=(20, 11))

    # 绘制历史价格
    ax.semilogy(daily.index, daily['close'], color='#f0883e', linewidth=1.2, alpha=0.9, label='历史价格')

    # 第四次减半: 2024-04-19
    halving_4 = datetime(2024, 4, 19)

    # 基于历史周期模式的预测参数
    # 周期2: 减半后517天到达峰值, 涨幅约30x
    # 周期3: 减半后546天到达峰值, 涨幅约8x（相对减半价）
    # 趋势: 每周期峰值倍数递减

    # 从第四次减半获取基准价格
    halving_price = daily.loc[daily.index.asof(halving_4), 'close']

    # 预测场景
    last_date = daily.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), end=halving_4 + timedelta(days=1460), freq='D')
    last_price = daily['close'].iloc[-1]

    # 基于周期分析的三种预测路径
    # 使用减半后的天数作为变量，使用对数增长+周期回归模型
    days_from_halving = np.array([(d - halving_4).days for d in future_dates])

    # 模型: 基于历史周期的三种情景
    # 情景1 (乐观): 类似周期2-3的上涨模式，峰值在减半后500-550天
    # 情景2 (基准): 考虑递减效应，涨幅为前一周期的40-50%
    # 情景3 (保守): 机构化市场，更温和的涨幅

    def cycle_model(days, peak_day, peak_mult, halving_price, decay_after_peak=0.6):
        """简化的周期价格模型"""
        prices = []
        for d in days:
            if d <= 0:
                prices.append(halving_price)
            elif d <= peak_day:
                # 上升阶段: 指数增长
                progress = d / peak_day
                growth = halving_price * (peak_mult ** progress)
                prices.append(growth)
            else:
                # 下降阶段: 从峰值回调
                peak_price = halving_price * peak_mult
                days_after_peak = d - peak_day
                decay = np.exp(-days_after_peak / 500) * decay_after_peak + (1 - decay_after_peak)
                prices.append(peak_price * decay)
        return np.array(prices)

    # 三种情景
    scenarios = [
        ("乐观情景", 520, 4.0, '#7ee787', '--'),   # 峰值4x, ~$260K
        ("基准情景", 550, 2.5, '#79c0ff', '-'),     # 峰值2.5x, ~$160K
        ("保守情景", 480, 1.8, '#ff7b72', '--'),     # 峰值1.8x, ~$115K
    ]

    for name, peak_day, peak_mult, color, ls in scenarios:
        pred_prices = cycle_model(days_from_halving, peak_day, peak_mult, halving_price)

        # 平滑连接: 从最后真实价格渐变到模型价格
        transition_days = 30
        if len(pred_prices) > transition_days:
            for i in range(min(transition_days, len(pred_prices))):
                alpha = i / transition_days
                pred_prices[i] = last_price * (1 - alpha) + pred_prices[i] * alpha

        ax.semilogy(future_dates, pred_prices, color=color, linewidth=2, linestyle=ls, alpha=0.8, label=f'{name} (峰值≈${halving_price*peak_mult:,.0f})')

        # 标注预测峰值
        peak_date = halving_4 + timedelta(days=peak_day)
        peak_price = halving_price * peak_mult
        if peak_date > last_date:
            ax.plot(peak_date, peak_price, '*', color=color, markersize=15, zorder=5)
            ax.annotate(f'${peak_price:,.0f}\n({peak_date.strftime("%Y-%m")})',
                       xy=(peak_date, peak_price),
                       xytext=(15, 15), textcoords='offset points',
                       fontsize=9, color=color, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor=color, alpha=0.8))

    # 标注预测区间
    ax.axvspan(last_date, future_dates[-1], alpha=0.04, color='#d2a8ff')
    ax.axvline(x=last_date, color='#e6edf3', linestyle='-', alpha=0.5, linewidth=1)
    ax.text(last_date + timedelta(days=10), daily['close'].min() * 1.5, '← 预测区间 →',
            fontsize=12, color='#d2a8ff', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor='#d2a8ff', alpha=0.8))

    # 标注减半事件
    for name, date in HALVINGS.items():
        if date >= daily.index[0]:
            ax.axvline(x=date, color='#d2a8ff', linestyle=':', alpha=0.4)

    # 下一次预计减半
    next_halving = datetime(2028, 4, 1)
    ax.axvline(x=next_halving, color='#ffa657', linestyle=':', alpha=0.4)
    ax.text(next_halving, daily['close'].max() * 0.5, '预计第五次减半\n~2028年4月',
            fontsize=9, color='#ffa657', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor='#ffa657', alpha=0.8))

    ax.set_title('BTC/USDT 基于四年周期的价格预测模型', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('价格 (USDT, 对数尺度)', fontsize=12)
    ax.set_xlabel('日期', fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.legend(loc='upper left', fontsize=11, facecolor='#21262d', edgecolor='#30363d')
    ax.grid(True, alpha=0.15, color='#30363d')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/10_prediction_model.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ 图表10已生成: {path}")
    return path


def chart11_yearly_performance(daily):
    """图表11: 年度表现统计"""
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    daily['year'] = daily.index.year

    # 1. 年度收益率
    ax = axes[0]
    yearly_data = []
    for year in sorted(daily['year'].unique()):
        year_df = daily[daily['year'] == year]
        if len(year_df) > 1:
            ret = (year_df['close'].iloc[-1] / year_df['close'].iloc[0] - 1) * 100
            yearly_data.append((year, ret))

    years_list = [y[0] for y in yearly_data]
    returns_list = [y[1] for y in yearly_data]
    colors_yr = ['#7ee787' if v > 0 else '#ff4444' for v in returns_list]

    bars = ax.bar(range(len(years_list)), returns_list, color=colors_yr, alpha=0.8)
    ax.set_xticks(range(len(years_list)))
    ax.set_xticklabels(years_list, rotation=45)
    for i, (y, r) in enumerate(zip(years_list, returns_list)):
        ax.text(i, r + (5 if r > 0 else -15), f'{r:.0f}%', ha='center', fontsize=8, fontweight='bold',
                color='#e6edf3')
    ax.set_title('年度收益率', fontsize=14, fontweight='bold')
    ax.set_ylabel('收益率 (%)')
    ax.axhline(y=0, color='#484f58', linestyle='-', alpha=0.5)
    ax.grid(True, alpha=0.15, color='#30363d')

    # 2. 年度最大回撤
    ax = axes[1]
    max_drawdowns = []
    for year in sorted(daily['year'].unique()):
        year_df = daily[daily['year'] == year]
        cummax = year_df['high'].cummax()
        dd = ((year_df['close'] - cummax) / cummax * 100).min()
        max_drawdowns.append((year, dd))

    years_dd = [y[0] for y in max_drawdowns]
    dd_vals = [y[1] for y in max_drawdowns]

    ax.bar(range(len(years_dd)), dd_vals, color='#ff4444', alpha=0.7)
    ax.set_xticks(range(len(years_dd)))
    ax.set_xticklabels(years_dd, rotation=45)
    for i, (y, d) in enumerate(zip(years_dd, dd_vals)):
        ax.text(i, d - 3, f'{d:.0f}%', ha='center', fontsize=8, fontweight='bold', color='#e6edf3')
    ax.set_title('年度最大回撤', fontsize=14, fontweight='bold')
    ax.set_ylabel('最大回撤 (%)')
    ax.grid(True, alpha=0.15, color='#30363d')

    # 3. 年度波动率
    ax = axes[2]
    vol_yearly = []
    for year in sorted(daily['year'].unique()):
        year_df = daily[daily['year'] == year]
        if len(year_df) > 10:
            vol = year_df['close'].pct_change().std() * np.sqrt(365) * 100
            vol_yearly.append((year, vol))

    years_vol = [y[0] for y in vol_yearly]
    vol_vals = [y[1] for y in vol_yearly]

    # 减半年份用不同颜色
    halving_set = {2020, 2024}
    colors_vol = ['#d2a8ff' if y in halving_set else '#79c0ff' for y in years_vol]

    ax.bar(range(len(years_vol)), vol_vals, color=colors_vol, alpha=0.7)
    ax.set_xticks(range(len(years_vol)))
    ax.set_xticklabels(years_vol, rotation=45)
    for i, (y, v) in enumerate(zip(years_vol, vol_vals)):
        ax.text(i, v + 1, f'{v:.0f}%', ha='center', fontsize=8, fontweight='bold', color='#e6edf3')
    ax.set_title('年化波动率 (紫色=减半年)', fontsize=14, fontweight='bold')
    ax.set_ylabel('年化波动率 (%)')
    ax.grid(True, alpha=0.15, color='#30363d')

    fig.suptitle('BTC/USDT 年度表现综合统计', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/11_yearly_performance.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ 图表11已生成: {path}")
    return path


def chart12_cycle_phase_analysis(daily):
    """图表12: 周期阶段分析（积累-上涨-分配-下跌）"""
    fig, ax = plt.subplots(figsize=(20, 10))

    # 定义各周期阶段
    phases = [
        # 周期3 (2016减半后)
        ("牛市上涨", datetime(2017, 8, 17), datetime(2017, 12, 17), '#7ee787'),
        ("见顶分配", datetime(2017, 12, 17), datetime(2018, 2, 6), '#ffa657'),
        ("熊市下跌", datetime(2018, 2, 6), datetime(2018, 12, 15), '#ff4444'),
        ("底部积累", datetime(2018, 12, 15), datetime(2019, 4, 1), '#79c0ff'),
        ("复苏反弹", datetime(2019, 4, 1), datetime(2019, 6, 26), '#7ee787'),
        ("再次调整", datetime(2019, 6, 26), datetime(2020, 3, 13), '#ffa657'),
        ("V型反转+减半", datetime(2020, 3, 13), datetime(2020, 5, 11), '#d2a8ff'),

        # 周期4 (2020减半后)
        ("牛市启动", datetime(2020, 5, 11), datetime(2021, 4, 14), '#7ee787'),
        ("中期调整", datetime(2021, 4, 14), datetime(2021, 7, 20), '#ffa657'),
        ("二次冲顶", datetime(2021, 7, 20), datetime(2021, 11, 10), '#7ee787'),
        ("熊市下跌", datetime(2021, 11, 10), datetime(2022, 11, 21), '#ff4444'),
        ("底部积累", datetime(2022, 11, 21), datetime(2023, 1, 13), '#79c0ff'),
        ("复苏上涨", datetime(2023, 1, 13), datetime(2024, 3, 14), '#7ee787'),
        ("减半前盘整", datetime(2024, 3, 14), datetime(2024, 4, 19), '#d2a8ff'),

        # 周期5 (2024减半后)
        ("减半后震荡", datetime(2024, 4, 19), datetime(2024, 10, 10), '#ffa657'),
        ("ETF驱动上涨", datetime(2024, 10, 10), daily.index[-1], '#7ee787'),
    ]

    ax.semilogy(daily.index, daily['close'], color='#e6edf3', linewidth=0.5, alpha=0.3)

    for label, start, end, color in phases:
        mask = (daily.index >= start) & (daily.index <= end)
        if mask.any():
            phase_data = daily.loc[mask]
            ax.semilogy(phase_data.index, phase_data['close'], color=color, linewidth=2, alpha=0.85)
            ax.axvspan(start, end, alpha=0.06, color=color)

            # 在区间中间标注
            mid_date = start + (end - start) / 2
            mid_price = phase_data['close'].iloc[len(phase_data)//2]
            ret = (phase_data['close'].iloc[-1] / phase_data['close'].iloc[0] - 1) * 100
            sign = '+' if ret > 0 else ''

    # 标注减半
    for name, date in HALVINGS.items():
        if date >= daily.index[0]:
            ax.axvline(x=date, color='#d2a8ff', linestyle='--', alpha=0.7, linewidth=2)
            ax.text(date, daily['close'].max() * 1.1, name, fontsize=9, color='#d2a8ff',
                    ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#21262d', edgecolor='#d2a8ff', alpha=0.8))

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#7ee787', linewidth=3, label='上涨/牛市'),
        Line2D([0], [0], color='#ffa657', linewidth=3, label='盘整/分配'),
        Line2D([0], [0], color='#ff4444', linewidth=3, label='下跌/熊市'),
        Line2D([0], [0], color='#79c0ff', linewidth=3, label='底部积累'),
        Line2D([0], [0], color='#d2a8ff', linewidth=3, label='减半事件'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, facecolor='#21262d', edgecolor='#30363d')

    ax.set_title('BTC/USDT 周期阶段划分分析（积累→上涨→分配→下跌）', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('价格 (USDT, 对数尺度)', fontsize=12)
    ax.set_xlabel('日期', fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.grid(True, alpha=0.15, color='#30363d')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/12_cycle_phase_analysis.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ 图表12已生成: {path}")
    return path


def generate_statistics(data):
    """生成统计数据用于报告"""
    daily = data['1d']
    weekly = data['1w']
    monthly = data['1M']

    stats = {}

    # 基本统计
    stats['total_days'] = len(daily)
    stats['date_range'] = f"{daily.index[0].strftime('%Y-%m-%d')} ~ {daily.index[-1].strftime('%Y-%m-%d')}"
    stats['current_price'] = daily['close'].iloc[-1]
    stats['ath'] = daily['high'].max()
    stats['ath_date'] = daily['high'].idxmax().strftime('%Y-%m-%d')
    stats['atl'] = daily['low'].min()
    stats['atl_date'] = daily['low'].idxmin().strftime('%Y-%m-%d')

    # 周期统计
    halving_3 = datetime(2020, 5, 11)
    halving_4 = datetime(2024, 4, 19)

    # 周期3 (2016-2020 减半间)
    cycle3_start = daily.index[0]  # 从币安数据开始
    cycle3_end = halving_3
    c3 = daily.loc[cycle3_start:cycle3_end]
    stats['cycle3_peak'] = c3['high'].max()
    stats['cycle3_peak_date'] = c3['high'].idxmax().strftime('%Y-%m-%d')
    stats['cycle3_trough'] = c3['low'].min()
    stats['cycle3_trough_date'] = c3['low'].idxmin().strftime('%Y-%m-%d')
    stats['cycle3_peak_to_trough'] = (stats['cycle3_trough'] - stats['cycle3_peak']) / stats['cycle3_peak'] * 100

    # 周期4 (2020-2024 减半间)
    c4 = daily.loc[halving_3:halving_4]
    stats['cycle4_peak'] = c4['high'].max()
    stats['cycle4_peak_date'] = c4['high'].idxmax().strftime('%Y-%m-%d')
    stats['cycle4_trough'] = c4['low'].min()
    stats['cycle4_trough_date'] = c4['low'].idxmin().strftime('%Y-%m-%d')
    halving3_price = daily.loc[daily.index.asof(halving_3), 'close']
    stats['cycle4_halving_to_peak'] = (stats['cycle4_peak'] - halving3_price) / halving3_price * 100
    stats['cycle4_peak_to_trough'] = (stats['cycle4_trough'] - stats['cycle4_peak']) / stats['cycle4_peak'] * 100

    # 当前周期 (2024减半后)
    c5 = daily.loc[halving_4:]
    stats['cycle5_peak'] = c5['high'].max()
    stats['cycle5_peak_date'] = c5['high'].idxmax().strftime('%Y-%m-%d')
    halving4_price = daily.loc[daily.index.asof(halving_4), 'close']
    stats['cycle5_halving_price'] = halving4_price
    stats['cycle5_current_mult'] = stats['current_price'] / halving4_price
    stats['cycle5_days_since_halving'] = (daily.index[-1] - halving_4).days

    # 年度统计
    daily['year'] = daily.index.year
    yearly_stats = []
    for year in sorted(daily['year'].unique()):
        yr = daily[daily['year'] == year]
        if len(yr) > 1:
            ret = (yr['close'].iloc[-1] / yr['close'].iloc[0] - 1) * 100
            vol = yr['close'].pct_change().std() * np.sqrt(365) * 100
            max_dd = ((yr['close'] - yr['high'].cummax()) / yr['high'].cummax() * 100).min()
            yearly_stats.append({
                'year': year,
                'open': yr['close'].iloc[0],
                'close': yr['close'].iloc[-1],
                'high': yr['high'].max(),
                'low': yr['low'].min(),
                'return': ret,
                'volatility': vol,
                'max_drawdown': max_dd,
                'avg_volume': yr['quote_volume'].mean(),
            })
    stats['yearly'] = yearly_stats

    # 月度统计
    monthly['returns'] = monthly['close'].pct_change() * 100
    stats['best_month'] = monthly['returns'].max()
    stats['best_month_date'] = monthly['returns'].idxmax().strftime('%Y-%m')
    stats['worst_month'] = monthly['returns'].min()
    stats['worst_month_date'] = monthly['returns'].idxmin().strftime('%Y-%m')
    stats['positive_months'] = (monthly['returns'] > 0).sum()
    stats['negative_months'] = (monthly['returns'] < 0).sum()
    stats['positive_month_ratio'] = stats['positive_months'] / (stats['positive_months'] + stats['negative_months']) * 100

    return stats


def main():
    print("=" * 60)
    print("BTC 四年周期理论验证 - 数据分析")
    print("=" * 60)

    data = load_data()
    daily = data['1d']
    h8 = data['8h']
    weekly = data['1w']
    monthly = data['1M']

    print(f"\n生成分析图表...")
    print("-" * 40)

    charts = []
    charts.append(chart1_price_overview_with_halvings(daily))
    charts.append(chart2_cycle_comparison_from_halving(daily))
    charts.append(chart3_drawdown_analysis(daily))
    charts.append(chart4_monthly_returns_heatmap(monthly))
    charts.append(chart5_volume_analysis(daily))
    charts.append(chart6_volatility_cycle(daily))
    charts.append(chart7_8h_kline_analysis(h8))
    charts.append(chart8_weekly_analysis(weekly))
    charts.append(chart9_moving_averages(daily))
    charts.append(chart10_prediction_model(daily))
    charts.append(chart11_yearly_performance(daily))
    charts.append(chart12_cycle_phase_analysis(daily))

    print(f"\n统计数据计算中...")
    stats = generate_statistics(data)

    # 保存统计数据
    import json
    with open('data/statistics.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    print(f"✓ 统计数据已保存: data/statistics.json")

    print(f"\n{'=' * 60}")
    print(f"所有图表生成完成! 共 {len(charts)} 张图表")
    print(f"图表目录: {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
