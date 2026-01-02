import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# 1. Stworzenie i przygotowanie danych
time = np.arange(0, 200, 0.1)
data = np.sin(time)

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1))
train_size = int(len(data_scaled) * 0.8)
train, test = data_scaled[0:train_size,:], data_scaled[train_size:len(data_scaled),:]

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 20
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 2. Budowa i trening modelu
model_ts = Sequential([ LSTM(50, input_shape=(look_back, 1)), Dense(1) ])
model_ts.compile(optimizer='adam', loss='mean_squared_error')
model_ts.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

# 3. Predykcja i wizualizacja
train_predict = model_ts.predict(X_train)
test_predict = model_ts.predict(X_test)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

plt.figure(figsize=(15,6))
plt.plot(scaler.inverse_transform(data_scaled), label='Oryginalny sygnał')
plt.plot(np.arange(look_back, len(train_predict)+look_back), train_predict, label='Predykcja na danych treningowych')
plt.plot(np.arange(len(train_predict)+(2*look_back)+1, len(data_scaled)-1), test_predict, label='Predykcja na danych testowych')
plt.title("Prognozowanie fali sinusoidalnej za pomocą LSTM")
plt.legend()
plt.show()


# --- Krok 1: Pobranie i przygotowanie danych ---
# Pobieramy dane historyczne dla akcji Google (GOOGL) z ostatnich kilku lat
df = yf.download('GOOGL', start='2015-01-01', end='2023-12-31')
data = df['Close'].values.reshape(-1, 1) # Bierzemy tylko ceny zamknięcia

# Skalowanie danych do zakresu [0, 1] - kluczowe dla sieci neuronowych
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Podział na zbiór treningowy i testowy
train_size = int(len(data_scaled) * 0.8)
train, test = data_scaled[0:train_size,:], data_scaled[train_size:len(data_scaled),:]

# Funkcja do tworzenia sekwencji (bez zmian)
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60  # Użyjemy 60 poprzednich dni do przewidzenia następnego
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# Reshape do formatu [próbki, kroki_czasowe, cechy] (bez zmian)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# --- Krok 2: Budowa i Trening Modelu LSTM (architektura bez zmian!) ---
model_stock = Sequential([
    LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
    LSTM(50),
    Dense(1)
])
model_stock.compile(optimizer='adam', loss='mean_squared_error')
model_stock.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# --- Krok 3: Predykcja i Wizualizacja ---
test_predict = model_stock.predict(X_test)

# Odwrócenie skalowania, aby zobaczyć realne ceny
test_predict = scaler.inverse_transform(test_predict)
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

# Wizualizacja wyników
plt.figure(figsize=(15, 7))
plt.plot(df.index[train_size+look_back+1:], y_test_real, color='blue', label='Rzeczywista cena akcji Google')
plt.plot(df.index[train_size+look_back+1:], test_predict, color='red', label='Przewidziana cena akcji Google')
plt.title('Prognoza cen akcji Google za pomocą LSTM')
plt.xlabel('Data')
plt.ylabel('Cena (USD)')
plt.legend()
plt.show()

