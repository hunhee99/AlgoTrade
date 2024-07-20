import time
import pyupbit
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging
import os

# 1. 데이터 수집 및 전처리
def get_data(ticker, interval='minute1', count=200):
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
    return df

def preprocess_data(data):
    data = data[['close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# 2. 학습 데이터셋 생성
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# 3. LSTM 모델 생성 및 훈련
def create_and_train_model(X_train, Y_train, time_step):
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, Y_train, batch_size=1, epochs=20)
    return model

# 4. 매매 실행 함수
def execute_trade(upbit, ticker, action, amount):
    try:
        if action == 'buy':
            upbit.buy_market_order(ticker, amount)
            logging.info(f"Buying {amount} of {ticker}")
        elif action == 'sell':
            upbit.sell_market_order(ticker, amount)
            logging.info(f"Selling {amount} of {ticker}")
    except Exception as e:
        logging.error(f"Error executing trade: {e}")

# 5. 실시간 매매 함수
def real_time_trading(ticker, model, scaler, time_step, access_key, secret_key, trade_amount):
    logging.basicConfig(filename='trading.log', level=logging.INFO)
    upbit = pyupbit.Upbit(access_key, secret_key)

    in_position = False
    buy_price = 0

    while True:
        data = get_data(ticker)
        scaled_data, _ = preprocess_data(data)
        
        last_time_steps = scaled_data[-time_step:]
        X_test = last_time_steps.reshape(1, time_step, 1)
        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]
        current_price = data['close'].iloc[-1]

        logging.info(f"Predicted price: {predicted_price}, Current price: {current_price}")

        try:
            if predicted_price > current_price * 1.002 and not in_position:
                # Buy
                buy_price = current_price
                execute_trade(upbit, ticker, 'buy', trade_amount)
                in_position = True
            elif predicted_price < current_price * 0.998 and in_position:
                # Sell
                execute_trade(upbit, ticker, 'sell', upbit.get_balance(ticker))
                in_position = False

            time.sleep(60)  # 1분마다 업데이트

        except Exception as e:
            logging.error(f"Error in trading logic: {e}")
            time.sleep(60)

# 메인 함수
def main():
    access_key = os.getenv("UPBIT_ACCESS_KEY")
    secret_key = os.getenv("UPBIT_SECRET_KEY")
    ticker = "KRW-BTC"
    time_step = 60
    trade_amount = 100000  # 거래 금액 (KRW)

    data = get_data(ticker)
    scaled_data, scaler = preprocess_data(data)
    X, Y = create_dataset(scaled_data, time_step)
    
    model = create_and_train_model(X, Y, time_step)
    
    real_time_trading(ticker, model, scaler, time_step, access_key, secret_key, trade_amount)

if __name__ == "__main__":
    main()
