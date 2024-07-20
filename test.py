import pyupbit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# 데이터 수집 및 전처리
def get_data(ticker, interval='minute1', count=20000):
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
    return df

# 기술적 지표 추가
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

# 매매 신호 생성
def generate_signals(data):
    data['Signal'] = 0
    data.loc[(data['MACD'] > data['Signal_Line']) & (data['RSI'] < 80), 'Signal'] = 1
    data.loc[(data['MACD'] < data['Signal_Line']) & (data['RSI'] > 20), 'Signal'] = -1
    data['Position'] = data['Signal'].diff()
    return data

# 시뮬레이션
def simulate_trading(data, initial_balance=1000000):
    balance = initial_balance
    btc_balance = 0
    position = None
    balances = []

    for i in range(len(data)):
        current_signal = data['Signal'].iloc[i]
        current_price = data['close'].iloc[i]

        if current_signal == 1 and position != 'long':  # 매수 신호
            if position == 'short':
                balance += btc_balance * current_price
                btc_balance = 0
            btc_balance = balance / current_price
            balance = 0
            position = 'long'
        elif current_signal == -1 and position != 'short':  # 매도 신호
            if position == 'long':
                balance += btc_balance * current_price
                btc_balance = 0
            btc_balance = balance / current_price
            balance = 0
            position = 'short'

        total_balance = balance + btc_balance * current_price
        balances.append(total_balance)

    data['Balance'] = balances
    return data

# 결과 시각화
def plot_results(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data['close'], label='Close Price')
    plt.plot(data['Balance'], label='Balance')
    buy_signals = data[data['Signal'] == 1]
    sell_signals = data[data['Signal'] == -1]
    plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='g', label='Buy Signal', alpha=1)
    plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='r', label='Sell Signal', alpha=1)
    plt.legend()
    plt.title('Trading Strategy Simulation')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()

# 메인 함수
def main():
    ticker = "KRW-BTC"
    data = get_data(ticker, interval='minute1', count=20000)
    data = preprocess_data(data)
    data = generate_signals(data)
    data = simulate_trading(data)
    plot_results(data)

if __name__ == "__main__":
    main()
