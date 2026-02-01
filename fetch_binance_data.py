#!/usr/bin/env python3
"""
从币安公开 API 获取 BTCUSDT 全量 K 线数据
支持日K、8小时K、周K、月K
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime

BASE_URL = "https://api.binance.com/api/v3/klines"

# 币安 K 线支持的时间间隔
INTERVALS = {
    "1d": "日K",
    "8h": "8小时K",
    "1w": "周K",
    "1M": "月K",
}

# BTCUSDT 上线时间约为 2017-08-17，但我们可以从更早获取（API会自动返回可用数据）
START_TIME = int(datetime(2017, 8, 17).timestamp() * 1000)  # 毫秒时间戳


def fetch_klines(symbol: str, interval: str, start_time: int, limit: int = 1000):
    """
    获取 K 线数据

    币安 API 文档:
    GET /api/v3/klines
    参数:
      - symbol: 交易对 (如 BTCUSDT)
      - interval: K线间隔 (1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M)
      - startTime: 开始时间(毫秒)
      - limit: 返回数量，默认500，最大1000

    返回数组格式:
    [
      [Open time, Open, High, Low, Close, Volume, Close time,
       Quote asset volume, Number of trades, Taker buy base volume,
       Taker buy quote volume, Ignore]
    ]
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "limit": limit,
    }

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_all_klines(symbol: str, interval: str):
    """获取某交易对某时间间隔的所有历史K线数据"""
    all_data = []
    current_start = START_TIME

    print(f"正在获取 {symbol} {INTERVALS.get(interval, interval)} 数据...")

    while True:
        try:
            data = fetch_klines(symbol, interval, current_start)
        except Exception as e:
            print(f"  请求出错: {e}, 重试...")
            time.sleep(2)
            continue

        if not data:
            break

        all_data.extend(data)

        # 下次请求从最后一条数据的收盘时间+1开始
        last_close_time = data[-1][6]
        current_start = last_close_time + 1

        print(f"  已获取 {len(all_data)} 条记录, 最新时间: {datetime.fromtimestamp(data[-1][0]/1000).strftime('%Y-%m-%d')}")

        # 如果返回数据不足1000条，说明已经到最新了
        if len(data) < 1000:
            break

        # 避免触发频率限制
        time.sleep(0.2)

    return all_data


def to_dataframe(raw_data):
    """将原始 K 线数据转换为 DataFrame"""
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(raw_data, columns=columns)

    # 类型转换
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                 "taker_buy_base", "taker_buy_quote"]:
        df[col] = df[col].astype(float)

    df["trades"] = df["trades"].astype(int)
    df.drop(columns=["ignore"], inplace=True)

    # 设置日期索引
    df.set_index("open_time", inplace=True)
    df.index.name = "date"

    return df


def main():
    os.makedirs("data", exist_ok=True)

    symbol = "BTCUSDT"

    for interval, name in INTERVALS.items():
        raw = fetch_all_klines(symbol, interval)
        df = to_dataframe(raw)

        filename = f"data/btcusdt_{interval}.csv"
        df.to_csv(filename)
        print(f"✓ {name} 数据已保存: {filename} ({len(df)} 条记录)")
        print(f"  时间范围: {df.index[0]} ~ {df.index[-1]}")
        print()

    print("所有数据获取完成!")


if __name__ == "__main__":
    main()
