import time
import pyupbit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

# 1. 데이터 수집 및 전처리
def get_data(ticker, interval='minute60', count=2000):
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
    return df

# 2. 기술적 지표 추가
def preprocess_data(data):
    data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['RSI'] = compute_rsi(data['close'])
    return data

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 3. 매매 신호 생성
def generate_signals(data):
    data['Signal'] = 0
    data.loc[(data['MACD'] > data['Signal_Line']) & (data['RSI'] < 70), 'Signal'] = 1
    data.loc[(data['MACD'] < data['Signal_Line']) & (data['RSI'] > 30), 'Signal'] = -1
    data['Position'] = data['Signal'].diff()
    return data

# 4. 시뮬레이션 실행
def simulate_trading(ticker, data):
    initial_balance = 10000000  # 10 million KRW
    balance = initial_balance
    position = 0  # 0 means no position, 1 means holding the asset
    num_coins = 0

    performance = []

    for index, row in data.iterrows():
        current_signal = row['Signal']
        current_price = row['close']

        if current_signal == 1 and position == 0:  # 매수 신호
            num_coins = balance / current_price
            balance = 0
            position = 1
        elif current_signal == -1 and position == 1:  # 매도 신호
            balance = num_coins * current_price
            num_coins = 0
            position = 0

        total_value = balance + num_coins * current_price
        performance.append(total_value)

    data['Performance'] = performance

    # 수익 그래프 출력
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Performance'], label='Portfolio Value')
    plt.title(f'Simulation for {ticker}')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value (KRW)')
    plt.legend()
    plt.show()

    final_balance = balance + num_coins * data.iloc[-1]['close']
    return final_balance

# 상위 10개 종목 분석 및 시뮬레이션
def main():
    tickers = pyupbit.get_tickers(fiat="KRW")
    tickers = tickers[:10]  # 상위 10개 종목 선택

    for ticker in tickers:
        data = get_data(ticker, interval='minute60', count=2000)  # 더 긴 기간의 데이터를 가져옴
        data = preprocess_data(data)
        data = generate_signals(data)
        final_balance = simulate_trading(ticker, data)
        print(f'Final balance for {ticker}: {final_balance}')

if __name__ == "__main__":
    main()
