import pyupbit
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# 1. 데이터 수집 및 전처리
def get_data(ticker, interval='minute1', count=2000):
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

# 4. 백테스트 함수
def backtest_strategy(ticker, time_step, initial_balance=1000000, fee_rate=0.0005):
    # 데이터 수집 및 전처리
    data = get_data(ticker, count=2000)  # 더 많은 데이터 수집
    scaled_data, scaler = preprocess_data(data)
    
    # 학습 데이터 생성
    X, Y = create_dataset(scaled_data, time_step)
    
    # 모델 학습
    model = create_and_train_model(X, Y, time_step)
    
    # 백테스트 시뮬레이션
    balance = initial_balance
    in_position = False
    buy_price = 0
    portfolio_value = []
    dates = data.index[time_step:-1]

    for i in range(time_step, len(scaled_data) - 1):
        last_time_steps = scaled_data[i-time_step:i]
        X_test = last_time_steps.reshape(1, time_step, 1)
        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]
        current_price = data['close'].iloc[i]

        # 예측 값과 현재 가격 출력
        print(f"Predicted price: {predicted_price}, Current price: {current_price}")

        try:
            # 매매 조건 재조정: 예측된 가격이 현재 가격보다 0.2% 이상 높은 경우에 매수, 낮은 경우에 매도
            if predicted_price > current_price * 1.002 and not in_position:
                # Buy
                buy_price = current_price * (1 + fee_rate)  # 매수 수수료 적용
                in_position = True
                print(f"Buying at {current_price}")
            elif predicted_price < current_price * 0.998 and in_position:
                # Sell
                balance = balance * (current_price / buy_price) * (1 - fee_rate)  # 매도 수수료 적용
                in_position = False
                print(f"Selling at {current_price}")
            
            # 포트폴리오 가치 계산
            if in_position:
                current_value = balance * (current_price / buy_price)
            else:
                current_value = balance

            # 포트폴리오 가치가 음수나 0 이하로 떨어지지 않도록 조정
            if current_value <= 0:
                current_value = 0.001  # 최소 값 설정

            portfolio_value.append(current_value)

        except Exception as e:
            print(f"Error in trading logic: {e}")
            portfolio_value.append(balance)

    # 최종 포트폴리오 가치 출력
    final_balance = portfolio_value[-1]
    print(f"Initial balance: {initial_balance}")
    print(f"Final balance: {final_balance}")
    print(f"Profit: {final_balance - initial_balance}")
    print(f"Return: {(final_balance / initial_balance - 1) * 100}%")

    # 포트폴리오 가치 그래프 출력
    plt.plot(dates, portfolio_value, label='Portfolio Value')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# 메인 함수
def main():
    ticker = "KRW-BTC"
    time_step = 60
    
    backtest_strategy(ticker, time_step)

if __name__ == "__main__":
    main()
