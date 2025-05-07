import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM

# Mendapatkan data saham SIDO
sidodata = yf.download('SIDO.JK', start='2020-01-01', end='2025-01-01')

# Menampilkan data
st.title('Pemantauan Pergerakan Saham SIDO')
st.subheader('Grafik Harga Saham SIDO')
st.line_chart(sidodata['Close'])

# Menyiapkan data untuk prediksi
data = sidodata[['Close']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[0:train_size], scaled_data[train_size:]

def create_dataset(data, window=60):
    X, y = [], []
    for i in range(len(data) - window - 1):
        X.append(data[i:(i + window), 0])
        y.append(data[i + window, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Prediksi harga saham
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Menampilkan hasil prediksi
st.subheader('Prediksi Harga Saham SIDO')
st.line_chart(predictions)
