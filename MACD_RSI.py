import time
import pyupbit
import pandas as pd
import numpy as np
import logging

# 1. 데이터 수집 및 전처리
def get_data(ticker, interval='minute1', count=200):
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

# 4. 매매 실행
def execute_trading_strategy(ticker, access_key, secret_key):
    # 로깅 설정
    logging.basicConfig(filename='trading.log', level=logging.INFO, format='%(asctime)s - %(message)s')

    upbit = pyupbit.Upbit(access_key, secret_key)
    position = None

    while True:
        try:
            # 실시간 데이터 수집 및 전처리
            data = get_data(ticker)
            data = preprocess_data(data)
            data = generate_signals(data)

            current_signal = data['Signal'].iloc[-1]

            if current_signal == 1 and position != 'long':  # 매수 신호
                if position == 'short':
                    upbit.sell_market_order(ticker, upbit.get_balance(ticker))
                    logging.info(f'Sell to close short position for {ticker}')
                upbit.buy_market_order(ticker, upbit.get_balance("KRW") * 0.9995)
                position = 'long'
                logging.info(f'Buy {ticker}')
            elif current_signal == -1 and position != 'short':  # 매도 신호
                if position == 'long':
                    upbit.sell_market_order(ticker, upbit.get_balance(ticker))
                    logging.info(f'Sell to close long position for {ticker}')
                upbit.sell_market_order(ticker, upbit.get_balance(ticker))
                position = 'short'
                logging.info(f'Sell {ticker}')

            time.sleep(60)  # 1분 간격으로 실행

        except Exception as e:
            logging.error(f'Error in trading strategy: {e}')
            time.sleep(60)

# 메인 함수
def main():
    access_key = "your-access-key"
    secret_key = "your-secret-key"
    ticker = "KRW-BTC"

    # 실시간 매매 전략 실행
    execute_trading_strategy(ticker, access_key, secret_key)

if __name__ == "__main__":
    main()
