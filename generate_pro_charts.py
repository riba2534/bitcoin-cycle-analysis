#!/usr/bin/env python3
"""
增强图表: Mayer Multiple + Power Law 对数回归
华尔街专家级补充分析
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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

HALVINGS = {
    "第三次减半": datetime(2020, 5, 11),
    "第四次减半": datetime(2024, 4, 19),
}


def chart13_mayer_multiple(daily):
    """图表13: Mayer Multiple 分析"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), height_ratios=[1.5, 1])

    ma200 = daily['close'].rolling(200).mean()
    mayer = daily['close'] / ma200

    # 上图: 价格 + MA200
    ax1.semilogy(daily.index, daily['close'], color='#f0883e', linewidth=1.2, alpha=0.9, label='BTC 价格')
    ax1.semilogy(daily.index, ma200, color='#79c0ff', linewidth=2, alpha=0.85, label='200日均线 (MA200)')

    # 着色: Mayer > 2.4 过热, < 0.7 超卖
    for i in range(1, len(daily)):
        if not np.isnan(mayer.iloc[i]):
            if mayer.iloc[i] > 2.4:
                ax1.axvspan(daily.index[i-1], daily.index[i], alpha=0.2, color='#ff4444', linewidth=0)
            elif mayer.iloc[i] < 0.7:
                ax1.axvspan(daily.index[i-1], daily.index[i], alpha=0.2, color='#7ee787', linewidth=0)

    for name, date in HALVINGS.items():
        ax1.axvline(x=date, color='#d2a8ff', linestyle='--', alpha=0.5, linewidth=1)

    ax1.set_title('BTC Mayer Multiple 估值分析 (红色=过热 Mayer>2.4, 绿色=超卖 Mayer<0.7)', fontsize=15, fontweight='bold', pad=15)
    ax1.set_ylabel('价格 (USDT)', fontsize=12)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.legend(loc='upper left', fontsize=11, facecolor='#21262d', edgecolor='#30363d')
    ax1.grid(True, alpha=0.15, color='#30363d')

    # 下图: Mayer Multiple 值
    ax2.plot(daily.index, mayer, color='#f0883e', linewidth=1, alpha=0.8)
    ax2.fill_between(daily.index, mayer, 1.0, where=mayer > 1, alpha=0.15, color='#7ee787')
    ax2.fill_between(daily.index, mayer, 1.0, where=mayer < 1, alpha=0.15, color='#ff4444')

    # 关键水平线
    levels = [
        (2.4, '过热线 (2.4)', '#ff4444', '--'),
        (1.5, '牛市参考 (1.5)', '#ffa657', ':'),
        (1.0, '均衡线 (1.0)', '#8b949e', '-'),
        (0.7, '超卖线 (0.7)', '#7ee787', '--'),
    ]
    for val, label, color, ls in levels:
        ax2.axhline(y=val, color=color, linestyle=ls, alpha=0.6, linewidth=1.5)
        ax2.text(daily.index[5], val + 0.05, label, fontsize=9, color=color, fontweight='bold')

    # 分位数标注
    mayer_clean = mayer.dropna()
    current = mayer_clean.iloc[-1]
    percentile = (mayer_clean < current).mean() * 100

    ax2.annotate(f'当前: {current:.3f}\n(P{percentile:.0f} 分位)',
                xy=(daily.index[-1], current),
                xytext=(-120, 40), textcoords='offset points',
                fontsize=11, color='#ff7b72', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#ff7b72', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#21262d', edgecolor='#ff7b72', alpha=0.9))

    for name, date in HALVINGS.items():
        ax2.axvline(x=date, color='#d2a8ff', linestyle='--', alpha=0.5, linewidth=1)

    ax2.set_ylabel('Mayer Multiple', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylim(0, 3.5)
    ax2.grid(True, alpha=0.15, color='#30363d')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/13_mayer_multiple.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ 图表13已生成: {path}")
    return path


def chart14_power_law(daily):
    """图表14: Power Law 对数回归通道"""
    fig, ax = plt.subplots(figsize=(20, 10))

    genesis = datetime(2009, 1, 3)
    daily['days_from_genesis'] = (daily.index - genesis).days

    log_price = np.log10(daily['close'].values)
    log_days = np.log10(daily['days_from_genesis'].values)

    mask = np.isfinite(log_price) & np.isfinite(log_days)
    coeffs = np.polyfit(log_days[mask], log_price[mask], 1)

    # Power Law 拟合线 + 上下通道
    all_days = daily['days_from_genesis'].values
    log_days_all = np.log10(all_days)
    fitted = 10 ** (coeffs[0] * log_days_all + coeffs[1])

    residuals = log_price[mask] - (coeffs[0] * log_days[mask] + coeffs[1])
    std_residual = np.std(residuals)

    upper_1 = 10 ** (coeffs[0] * log_days_all + coeffs[1] + 1.5 * std_residual)
    lower_1 = 10 ** (coeffs[0] * log_days_all + coeffs[1] - 1.5 * std_residual)
    upper_2 = 10 ** (coeffs[0] * log_days_all + coeffs[1] + 2.5 * std_residual)
    lower_2 = 10 ** (coeffs[0] * log_days_all + coeffs[1] - 2.5 * std_residual)

    # 绘制通道
    ax.fill_between(daily.index, lower_2, upper_2, alpha=0.08, color='#d2a8ff', label='±2.5σ 通道')
    ax.fill_between(daily.index, lower_1, upper_1, alpha=0.12, color='#79c0ff', label='±1.5σ 通道')
    ax.semilogy(daily.index, fitted, color='#7ee787', linewidth=2.5, label=f'Power Law (斜率={coeffs[0]:.2f})', alpha=0.9)

    # 价格
    ax.semilogy(daily.index, daily['close'], color='#f0883e', linewidth=1.2, alpha=0.9, label='BTC 价格')

    # 延伸预测到 2029
    future_start = daily.index[-1] + timedelta(days=1)
    future_dates = pd.date_range(start=future_start, end=datetime(2029, 1, 1), freq='D')
    future_days = np.array([(d - genesis).days for d in future_dates])
    future_log_days = np.log10(future_days)
    future_fitted = 10 ** (coeffs[0] * future_log_days + coeffs[1])
    future_upper1 = 10 ** (coeffs[0] * future_log_days + coeffs[1] + 1.5 * std_residual)
    future_lower1 = 10 ** (coeffs[0] * future_log_days + coeffs[1] - 1.5 * std_residual)
    future_upper2 = 10 ** (coeffs[0] * future_log_days + coeffs[1] + 2.5 * std_residual)
    future_lower2 = 10 ** (coeffs[0] * future_log_days + coeffs[1] - 2.5 * std_residual)

    ax.fill_between(future_dates, future_lower2, future_upper2, alpha=0.05, color='#d2a8ff')
    ax.fill_between(future_dates, future_lower1, future_upper1, alpha=0.08, color='#79c0ff')
    ax.semilogy(future_dates, future_fitted, color='#7ee787', linewidth=2, linestyle='--', alpha=0.6)

    # 预测区间分界线
    ax.axvline(x=daily.index[-1], color='#e6edf3', linestyle='-', alpha=0.3)
    ax.text(daily.index[-1] + timedelta(days=30), daily['close'].min() * 1.5, '预测区间 →',
            fontsize=10, color='#d2a8ff', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor='#d2a8ff', alpha=0.8))

    # 标注关键信息
    current_fitted = 10 ** (coeffs[0] * np.log10(all_days[-1]) + coeffs[1])
    ratio = daily['close'].iloc[-1] / current_fitted
    ax.annotate(f'当前: ${daily["close"].iloc[-1]:,.0f}\nPower Law: ${current_fitted:,.0f}\n比值: {ratio:.3f}',
               xy=(daily.index[-1], daily['close'].iloc[-1]),
               xytext=(-180, -60), textcoords='offset points',
               fontsize=10, color='#f0883e', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#f0883e', lw=1.5),
               bbox=dict(boxstyle='round,pad=0.4', facecolor='#21262d', edgecolor='#f0883e', alpha=0.9))

    # 2028 减半时 Power Law 预测
    halving5_date = datetime(2028, 4, 1)
    halving5_days = (halving5_date - genesis).days
    halving5_fitted = 10 ** (coeffs[0] * np.log10(halving5_days) + coeffs[1])
    halving5_upper = 10 ** (coeffs[0] * np.log10(halving5_days) + coeffs[1] + 1.5 * std_residual)
    ax.axvline(x=halving5_date, color='#ffa657', linestyle=':', alpha=0.4)
    ax.annotate(f'预计第五次减半 (~2028-04)\nPower Law: ${halving5_fitted:,.0f}\n+1.5σ: ${halving5_upper:,.0f}',
               xy=(halving5_date, halving5_fitted),
               xytext=(-20, 50), textcoords='offset points',
               fontsize=9, color='#ffa657', fontweight='bold', ha='center',
               arrowprops=dict(arrowstyle='->', color='#ffa657', lw=1),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor='#ffa657', alpha=0.8))

    for name, date in HALVINGS.items():
        ax.axvline(x=date, color='#d2a8ff', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_title('BTC Power Law 对数回归通道 (基于创世区块天数)', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('价格 (USDT, 对数尺度)', fontsize=12)
    ax.set_xlabel('日期', fontsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.legend(loc='upper left', fontsize=11, facecolor='#21262d', edgecolor='#30363d')
    ax.grid(True, alpha=0.15, color='#30363d')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/14_power_law_regression.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ 图表14已生成: {path}")
    return path


def chart15_risk_metrics(daily):
    """图表15: 滚动风险指标仪表盘"""
    fig, axes = plt.subplots(3, 1, figsize=(20, 15), height_ratios=[1, 1, 1])

    daily['returns'] = daily['close'].pct_change()
    ma200 = daily['close'].rolling(200).mean()

    # 1. 滚动 Sharpe Ratio (90日)
    ax = axes[0]
    rolling_ret = daily['returns'].rolling(90).mean() * 365
    rolling_vol = daily['returns'].rolling(90).std() * np.sqrt(365)
    rolling_sharpe = (rolling_ret - 0.04) / rolling_vol

    ax.plot(daily.index, rolling_sharpe, color='#79c0ff', linewidth=1.2, alpha=0.9)
    ax.fill_between(daily.index, rolling_sharpe, 0, where=rolling_sharpe > 0, alpha=0.15, color='#7ee787')
    ax.fill_between(daily.index, rolling_sharpe, 0, where=rolling_sharpe < 0, alpha=0.15, color='#ff4444')
    ax.axhline(y=0, color='#484f58', linestyle='-', alpha=0.5)
    ax.axhline(y=1, color='#7ee787', linestyle='--', alpha=0.3, label='Sharpe=1 (良好)')
    ax.axhline(y=-1, color='#ff4444', linestyle='--', alpha=0.3, label='Sharpe=-1')

    for name, date in HALVINGS.items():
        ax.axvline(x=date, color='#d2a8ff', linestyle='--', alpha=0.4)

    ax.set_title('90日滚动 Sharpe Ratio', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend(loc='upper right', fontsize=9, facecolor='#21262d', edgecolor='#30363d')
    ax.grid(True, alpha=0.15, color='#30363d')
    ax.set_ylim(-4, 6)

    # 2. 滚动最大回撤 (90日)
    ax = axes[1]
    rolling_dd = pd.Series(index=daily.index, dtype=float)
    for i in range(90, len(daily)):
        window = daily['close'].iloc[i-90:i+1]
        peak = window.cummax()
        dd = ((window - peak) / peak).min()
        rolling_dd.iloc[i] = dd * 100

    ax.plot(daily.index, rolling_dd, color='#ff7b72', linewidth=1, alpha=0.9)
    ax.fill_between(daily.index, rolling_dd, 0, alpha=0.2, color='#ff4444')
    ax.axhline(y=-20, color='#ffa657', linestyle='--', alpha=0.4, label='-20% 警戒线')
    ax.axhline(y=-50, color='#ff4444', linestyle='--', alpha=0.4, label='-50% 危机线')

    for name, date in HALVINGS.items():
        ax.axvline(x=date, color='#d2a8ff', linestyle='--', alpha=0.4)

    ax.set_title('90日滚动最大回撤', fontsize=14, fontweight='bold')
    ax.set_ylabel('最大回撤 (%)')
    ax.legend(loc='lower left', fontsize=9, facecolor='#21262d', edgecolor='#30363d')
    ax.grid(True, alpha=0.15, color='#30363d')

    # 3. 回撤恢复时间分析
    ax = axes[2]
    # 计算从ATH的恢复天数
    daily['ath'] = daily['high'].cummax()
    daily['drawdown_pct'] = (daily['close'] - daily['ath']) / daily['ath'] * 100

    # 计算每个点距离上一次ATH的天数
    days_since_ath = []
    last_ath_idx = 0
    for i in range(len(daily)):
        if daily['close'].iloc[i] >= daily['ath'].iloc[i] * 0.99:  # 接近ATH
            last_ath_idx = i
        days_since_ath.append(i - last_ath_idx)

    ax.bar(daily.index, days_since_ath, color='#ffa657', alpha=0.5, width=1)

    # 标注主要恢复
    ax.annotate('2017峰值→恢复\n1095天', xy=(datetime(2020, 12, 1), 1050),
               fontsize=9, color='#ffa657', ha='center', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='#21262d', edgecolor='#ffa657', alpha=0.8))
    ax.annotate('2021峰值→恢复\n852天', xy=(datetime(2024, 3, 1), 800),
               fontsize=9, color='#ffa657', ha='center', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='#21262d', edgecolor='#ffa657', alpha=0.8))

    for name, date in HALVINGS.items():
        ax.axvline(x=date, color='#d2a8ff', linestyle='--', alpha=0.4)

    ax.set_title('距离上一次 ATH 的天数 (水下时间)', fontsize=14, fontweight='bold')
    ax.set_ylabel('天数')
    ax.set_xlabel('日期', fontsize=12)
    ax.grid(True, alpha=0.15, color='#30363d')

    fig.suptitle('BTC/USDT 专业风险指标仪表盘', fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/15_risk_metrics_dashboard.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ 图表15已生成: {path}")
    return path


def main():
    daily = pd.read_csv('data/btcusdt_1d.csv', index_col='date', parse_dates=True)
    print("生成增强版专业图表...")
    chart13_mayer_multiple(daily)
    chart14_power_law(daily)
    chart15_risk_metrics(daily)
    print("增强图表生成完成!")


if __name__ == "__main__":
    main()
