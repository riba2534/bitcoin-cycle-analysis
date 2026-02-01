#!/usr/bin/env python3
"""
后市走势深度预测图表
基于周期映射 + 历史统计概率
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


def chart16_cycle_mapping_forecast(daily):
    """图表16: 周期3映射预测 + 递减修正"""
    fig, ax = plt.subplots(figsize=(20, 11))

    c3_peak = datetime(2021, 11, 10)
    c3_peak_price = 69000.0
    c4_peak = daily['high'].idxmax()
    c4_peak_price = float(daily['high'].max())
    current_date = daily.index[-1]

    # --- 绘制周期4实际价格（峰后） ---
    c4_after = daily.loc[c4_peak:]
    days_since_peak = (c4_after.index - c4_peak).days
    ax.plot(days_since_peak, c4_after['close'], color='#f0883e', linewidth=2.5, label='周期4 实际价格', zorder=5)
    current_day = (current_date - c4_peak).days

    # --- 绘制周期3实际路径（对齐并缩放） ---
    c3_after = daily.loc[c3_peak:c3_peak + timedelta(days=900)]
    c3_days = (c3_after.index - c3_peak).days
    c3_scaled = c4_peak_price * (c3_after['close'].values / c3_peak_price)
    ax.plot(c3_days, c3_scaled, color='#79c0ff', linewidth=1.5, alpha=0.7, linestyle='--', label='周期3 映射路径 (等比例缩放)')

    # --- 递减修正路径 (回撤幅度按0.5x系数缩放) ---
    decay_factor = 0.5  # 本轮回撤幅度预计为上轮的50%
    c3_dd_pct = (c3_after['close'].values / c3_peak_price - 1)  # 周期3回撤百分比
    c3_dd_scaled = c3_dd_pct * decay_factor  # 缩放后的回撤
    c3_moderated = c4_peak_price * (1 + c3_dd_scaled)
    ax.plot(c3_days, c3_moderated, color='#7ee787', linewidth=2, alpha=0.8, linestyle='-', label=f'递减修正路径 (回撤×{decay_factor})')

    # --- 标注关键价格水平 ---
    levels = [
        (c4_peak_price, 'ATH $126,200', '#ff7b72', 0),
        (103956, 'MA200 ~$104,000', '#79c0ff', 0),
        (93112, 'Power Law ~$93,000', '#7ee787', 0),
        (63818, '减半价 $63,818', '#d2a8ff', 0),
        (69000, '前轮ATH $69,000', '#ffa657', 0),
    ]
    for price, label, color, offset in levels:
        ax.axhline(y=price, color=color, linestyle=':', alpha=0.4, linewidth=1)
        ax.text(880, price * 1.02, label, fontsize=9, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#21262d', edgecolor=color, alpha=0.8))

    # --- 标注当前位置 ---
    ax.plot(current_day, daily['close'].iloc[-1], 'o', color='#f0883e', markersize=12, zorder=6)
    ax.annotate(f'当前: ${daily["close"].iloc[-1]:,.0f}\n(峰后 {current_day} 天)',
               xy=(current_day, daily['close'].iloc[-1]),
               xytext=(-100, 40), textcoords='offset points',
               fontsize=11, color='#f0883e', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#f0883e', lw=2),
               bbox=dict(boxstyle='round,pad=0.4', facecolor='#21262d', edgecolor='#f0883e', alpha=0.9))

    # --- 标注预测区间关键点 ---
    # 周期3映射底部
    c3_trough_day = 376
    c3_mapped_bottom = c4_peak_price * (15476 / c3_peak_price)
    ax.plot(c3_trough_day, c3_mapped_bottom, 'v', color='#79c0ff', markersize=12, zorder=5)
    ax.annotate(f'周期3映射底部\n${c3_mapped_bottom:,.0f} (Day {c3_trough_day})',
               xy=(c3_trough_day, c3_mapped_bottom),
               xytext=(30, -30), textcoords='offset points',
               fontsize=9, color='#79c0ff', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#79c0ff'),
               bbox=dict(boxstyle='round,pad=0.2', facecolor='#21262d', edgecolor='#79c0ff', alpha=0.8))

    # 递减修正底部
    moderated_trough_idx = np.argmin(c3_moderated[:500])
    moderated_trough_day = c3_days[moderated_trough_idx]
    moderated_trough_price = c3_moderated[moderated_trough_idx]
    ax.plot(moderated_trough_day, moderated_trough_price, '^', color='#7ee787', markersize=12, zorder=5)
    ax.annotate(f'递减修正底部\n${moderated_trough_price:,.0f} (Day {moderated_trough_day})',
               xy=(moderated_trough_day, moderated_trough_price),
               xytext=(30, 30), textcoords='offset points',
               fontsize=9, color='#7ee787', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#7ee787'),
               bbox=dict(boxstyle='round,pad=0.2', facecolor='#21262d', edgecolor='#7ee787', alpha=0.8))

    # 预测区域分界
    ax.axvline(x=current_day, color='#e6edf3', linestyle='-', alpha=0.3)
    ax.axvspan(current_day, 900, alpha=0.04, color='#d2a8ff')

    ax.set_title('周期3→周期4 映射预测: 峰后走势对比 (含递减修正)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('峰后天数', fontsize=13)
    ax.set_ylabel('价格 (USDT)', fontsize=13)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.legend(loc='upper right', fontsize=11, facecolor='#21262d', edgecolor='#30363d')
    ax.grid(True, alpha=0.15, color='#30363d')
    ax.set_xlim(-10, 900)
    ax.set_ylim(0, c4_peak_price * 1.1)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/16_cycle_mapping_forecast.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ 图表16已生成: {path}")


def chart17_probability_cones(daily):
    """图表17: 概率锥 - 基于历史波动率的未来价格区间"""
    fig, ax = plt.subplots(figsize=(20, 11))

    current_price = daily['close'].iloc[-1]
    current_date = daily.index[-1]

    # 绘制历史价格（近2年）
    hist = daily.loc[current_date - timedelta(days=730):]
    ax.semilogy(hist.index, hist['close'], color='#f0883e', linewidth=1.5, alpha=0.9, label='历史价格')

    # 计算历史波动率参数
    returns = daily['close'].pct_change().dropna()
    daily_vol = returns.std()
    daily_drift = returns.mean()

    # 生成概率锥 (1σ, 2σ)
    forecast_days = 730  # 预测2年
    forecast_dates = pd.date_range(start=current_date, periods=forecast_days, freq='D')
    t = np.arange(1, forecast_days + 1)

    # 对数正态模型
    log_drift = (daily_drift - 0.5 * daily_vol**2)
    log_vol = daily_vol

    median_path = current_price * np.exp(log_drift * t)
    upper_1s = current_price * np.exp((log_drift + 1 * log_vol * np.sqrt(t / t)) * t + 1 * log_vol * np.sqrt(t))
    lower_1s = current_price * np.exp((log_drift - 1 * log_vol * np.sqrt(t / t)) * t - 1 * log_vol * np.sqrt(t))
    upper_2s = current_price * np.exp(log_drift * t + 2 * log_vol * np.sqrt(t))
    lower_2s = current_price * np.exp(log_drift * t - 2 * log_vol * np.sqrt(t))

    # 简化: 使用布朗运动
    upper_1s = current_price * np.exp(log_drift * t + 1 * daily_vol * np.sqrt(t))
    lower_1s = current_price * np.exp(log_drift * t - 1 * daily_vol * np.sqrt(t))
    upper_2s = current_price * np.exp(log_drift * t + 2 * daily_vol * np.sqrt(t))
    lower_2s = current_price * np.exp(log_drift * t - 2 * daily_vol * np.sqrt(t))

    # 绘制概率锥
    ax.fill_between(forecast_dates, lower_2s, upper_2s, alpha=0.08, color='#d2a8ff', label='95% 置信区间 (±2σ)')
    ax.fill_between(forecast_dates, lower_1s, upper_1s, alpha=0.15, color='#79c0ff', label='68% 置信区间 (±1σ)')
    ax.semilogy(forecast_dates, median_path, color='#7ee787', linewidth=2, linestyle='--', label='中位数路径 (漂移)', alpha=0.8)

    # 标注关键时间点的价格区间
    key_days = [90, 180, 365, 730]
    for d in key_days:
        if d < forecast_days:
            date = forecast_dates[d-1]
            med = median_path[d-1]
            u1 = upper_1s[d-1]
            l1 = lower_1s[d-1]
            u2 = upper_2s[d-1]
            l2 = lower_2s[d-1]
            ax.plot([date, date], [l1, u1], color='#79c0ff', linewidth=3, alpha=0.6)
            label_text = f'{d}天后\n68%: ${l1:,.0f}-${u1:,.0f}\n95%: ${l2:,.0f}-${u2:,.0f}'
            ax.annotate(label_text,
                       xy=(date, med),
                       xytext=(15, 0), textcoords='offset points',
                       fontsize=8, color='#e6edf3',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor='#484f58', alpha=0.9))

    # 标注关键水平
    for price, label, color in [
        (126200, 'ATH', '#ff7b72'),
        (103956, 'MA200', '#79c0ff'),
        (63818, '减半价', '#d2a8ff'),
    ]:
        ax.axhline(y=price, color=color, linestyle=':', alpha=0.3)
        ax.text(hist.index[0] + timedelta(days=5), price * 1.03, label, fontsize=8, color=color)

    ax.set_title('BTC/USDT 概率锥预测 (基于历史波动率的对数正态模型)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('日期', fontsize=13)
    ax.set_ylabel('价格 (USDT, 对数尺度)', fontsize=13)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.legend(loc='upper left', fontsize=10, facecolor='#21262d', edgecolor='#30363d')
    ax.grid(True, alpha=0.15, color='#30363d')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/17_probability_cones.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ 图表17已生成: {path}")


def chart18_drawdown_scenario_timeline(daily):
    """图表18: 情景时间线图"""
    fig, ax = plt.subplots(figsize=(20, 10))

    c4_peak = daily['high'].idxmax()
    c4_peak_price = float(daily['high'].max())
    current_date = daily.index[-1]
    current_price = float(daily['close'].iloc[-1])

    # 绘制已实现价格
    recent = daily.loc[datetime(2024, 1, 1):]
    ax.plot(recent.index, recent['close'], color='#f0883e', linewidth=2, label='实际价格', zorder=5)

    # 三种情景路径
    scenarios = {
        '情景A: 温和调整 (-50%)': {
            'color': '#7ee787',
            'bottom_dd': -50,
            'bottom_date': datetime(2026, 8, 1),
            'recovery_date': datetime(2027, 6, 1),
            'final_date': datetime(2028, 4, 1),
        },
        '情景B: 标准熊市 (-60%)': {
            'color': '#79c0ff',
            'bottom_dd': -60,
            'bottom_date': datetime(2026, 10, 15),
            'recovery_date': datetime(2027, 12, 1),
            'final_date': datetime(2028, 4, 1),
        },
        '情景C: 深度熊市 (-70%)': {
            'color': '#ff7b72',
            'bottom_dd': -70,
            'bottom_date': datetime(2027, 2, 1),
            'recovery_date': datetime(2028, 6, 1),
            'final_date': datetime(2028, 10, 1),
        },
    }

    for name, s in scenarios.items():
        bottom_price = c4_peak_price * (1 + s['bottom_dd'] / 100)

        # 构建路径: 当前 → 底部 → 恢复
        path_dates = [current_date, s['bottom_date'], s['recovery_date'], s['final_date']]
        path_prices = [current_price, bottom_price, bottom_price * 1.8, c4_peak_price * 0.9]

        # 使用三次插值平滑
        from scipy.interpolate import CubicSpline
        path_days = [(d - current_date).days for d in path_dates]
        cs = CubicSpline(path_days, path_prices, bc_type='natural')

        interp_days = np.arange(0, path_days[-1] + 1)
        interp_dates = [current_date + timedelta(days=int(d)) for d in interp_days]
        interp_prices = cs(interp_days)
        interp_prices = np.maximum(interp_prices, bottom_price * 0.95)

        ax.plot(interp_dates, interp_prices, color=s['color'], linewidth=2, linestyle='--', alpha=0.8, label=name)

        # 标注底部
        ax.plot(s['bottom_date'], bottom_price, 'v', color=s['color'], markersize=10, zorder=5)
        ax.annotate(f'${bottom_price:,.0f}\n({s["bottom_date"].strftime("%Y-%m")})',
                   xy=(s['bottom_date'], bottom_price),
                   xytext=(0, -35), textcoords='offset points',
                   fontsize=8, color=s['color'], ha='center', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='#21262d', edgecolor=s['color'], alpha=0.8))

    # 标注关键水平
    key_levels = [
        (c4_peak_price, 'ATH $126,200', '#ff7b72'),
        (103956, 'MA200 $104K', '#79c0ff'),
        (93112, 'Power Law $93K', '#7ee787'),
        (63818, '减半价 $63.8K', '#d2a8ff'),
    ]
    for price, label, color in key_levels:
        ax.axhline(y=price, color=color, linestyle=':', alpha=0.3)

    # 预计下次减半
    ax.axvline(x=datetime(2028, 4, 1), color='#d2a8ff', linestyle='--', alpha=0.5)
    ax.text(datetime(2028, 4, 1), c4_peak_price * 1.03, '预计第五次减半', fontsize=10, color='#d2a8ff',
            ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor='#d2a8ff', alpha=0.8))

    ax.set_title('BTC/USDT 后市三种情景时间线 (基于历史周期模式)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('日期', fontsize=13)
    ax.set_ylabel('价格 (USDT)', fontsize=13)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.legend(loc='upper right', fontsize=11, facecolor='#21262d', edgecolor='#30363d')
    ax.grid(True, alpha=0.15, color='#30363d')
    ax.set_xlim(datetime(2024, 1, 1), datetime(2028, 12, 1))
    ax.set_ylim(0, c4_peak_price * 1.15)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/18_scenario_timeline.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"✓ 图表18已生成: {path}")


def main():
    daily = pd.read_csv('data/btcusdt_1d.csv', index_col='date', parse_dates=True)
    print("生成后市预测图表...")
    chart16_cycle_mapping_forecast(daily)
    chart17_probability_cones(daily)
    chart18_drawdown_scenario_timeline(daily)
    print("后市预测图表生成完成!")


if __name__ == "__main__":
    main()
