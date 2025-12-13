# --- Import niezbędnych bibliotek ---
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Ustawienie estetyki wykresów ---
sns.set_theme(style="whitegrid")

# --- Krok 1: Generowanie Syntetycznych Danych ---
# W realnym świecie te dane pochodziłyby z plików CSV lub z bazy danych.
# Tutaj wygenerujemy je, aby przykład był w pełni powtarzalny.

def generate_time_series_data(n_samples, seq_length, n_features):
    """Generuje dane dla 3 klas: 'Stanie', 'Chodzenie', 'Bieganie'."""
    X = []
    y = []
    
    for _ in range(n_samples):
        class_id = np.random.randint(3) # Losowa klasa 0, 1, lub 2
        
        if class_id == 0: # Klasa 0: Stanie (mały szum wokół zera)
            sequence = np.random.randn(seq_length, n_features) * 0.1
        elif class_id == 1: # Klasa 1: Chodzenie (cykliczny wzorzec o umiarkowanej amplitudzie)
            time = np.linspace(0, 4 * np.pi, seq_length)
            sequence = np.zeros((seq_length, n_features))
            sequence[:, 0] = np.sin(time) + np.random.randn(seq_length) * 0.2
            sequence[:, 1] = np.cos(time) + np.random.randn(seq_length) * 0.2
            sequence[:, 2] = np.sin(time/2) + np.random.randn(seq_length) * 0.2
        else: # Klasa 2: Bieganie (cykliczny wzorzec o dużej amplitudzie i częstotliwości)
            time = np.linspace(0, 8 * np.pi, seq_length)
            sequence = np.zeros((seq_length, n_features))
            sequence[:, 0] = 2 * np.sin(time) + np.random.randn(seq_length) * 0.3
            sequence[:, 1] = 2 * np.cos(time) + np.random.randn(seq_length) * 0.3
            sequence[:, 2] = 2 * np.sin(time/2) + np.random.randn(seq_length) * 0.3
            
        X.append(sequence)
        y.append(class_id)
        
    return np.array(X), np.array(y)

# Parametry danych
N_SAMPLES = 5000
SEQ_LENGTH = 128  # 128 odczytów z sensora
N_FEATURES = 3    # Osie X, Y, Z
CLASS_NAMES = ['Stanie', 'Chodzenie', 'Bieganie']

X, y = generate_time_series_data(N_SAMPLES, SEQ_LENGTH, N_FEATURES)
print(f"Kształt danych X: {X.shape}") # -> (5000, 128, 3)
print(f"Kształt etykiet y: {y.shape}")   # -> (5000,)

# --- Wizualizacja przykładowych sygnałów ---
plt.figure(figsize=(15, 5))
for i, activity in enumerate(CLASS_NAMES):
    plt.subplot(1, 3, i + 1)
    # Znajdź pierwszy przykład danej aktywności
    sample_idx = np.where(y == i)[0][0]
    plt.plot(X[sample_idx])
    plt.title(f'Przykład dla aktywności: {activity}')
    plt.xlabel('Krok czasowy')
    plt.ylabel('Wartość z sensora')
plt.tight_layout()
plt.show()

# --- Krok 2: Przygotowanie Danych do Treningu ---
# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Skalowanie danych - ważne, choć nie tak krytyczne jak przy obrazach
# Skalujemy każdą z 3 cech (X, Y, Z) niezależnie
# Musimy zmienić kształt danych, aby pasował do Scalera
scaler = StandardScaler()
X_train_reshaped = X_train.reshape(-1, N_FEATURES)
scaler.fit(X_train_reshaped)

X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, N_FEATURES)).reshape(X_test.shape)

# Konwersja etykiet na one-hot encoding (mamy 3 klasy)
y_train_cat = keras.utils.to_categorical(y_train, num_classes=3)
y_test_cat = keras.utils.to_categorical(y_test, num_classes=3)

# --- Definicja nazwy pliku modelu ---
model_filename = "series_cnn_model.keras"

# --- GŁÓWNA LOGIKA: WCZYTAJ LUB TRENUJ ---
import os
# Sprawdzamy, czy plik z modelem już istnieje
if os.path.exists(model_filename):
    # --- ŚCIEŻKA 1: Plik istnieje - wczytujemy model ---
    print(f"--- Znaleziono istniejący model '{model_filename}'. Wczytywanie... ---")
    model_ts = keras.models.load_model(model_filename)
    print("Model został pomyślnie wczytany.")

else:
    # --- ŚCIEŻKA 2: Plik nie istnieje - definiujemy, kompilujemy i trenujemy model ---
    print(f"--- Nie znaleziono modelu '{model_filename}'. Rozpoczynanie nowego treningu... ---")


    # --- Krok 3: Budowa i Trening Modelu CNN 1D ---
    model_ts = Sequential([
        Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(SEQ_LENGTH, N_FEATURES)),
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(units=3, activation='softmax') # 3 neurony na wyjściu dla 3 klas
    ])


    # Kompilacja modelu
    model_ts.compile(optimizer='adam', 
                    loss='categorical_crossentropy', # Używamy dla klasyfikacji wieloklasowej
                    metrics=['accuracy'])

    # Trening
    print("\n--- Rozpoczynanie treningu modelu dla szeregów czasowych ---")
    history = model_ts.fit(X_train_scaled, y_train_cat, 
                        epochs=20, 
                        batch_size=64, 
                        validation_split=0.2)
    
    # Wizualizacja historii treningu
    pd.DataFrame(history.history).plot(figsize=(10, 6))
    plt.grid(True)
    plt.gca().set_ylim(0, 1.2) # Ustawienie limitów osi Y dla lepszej czytelności
    plt.title("Historia treningu modelu")
    plt.show()

    print(f"\nZapisywanie nowego modelu do pliku '{model_filename}'...")
    model_ts.save(model_filename)
    print("Model pomyślnie zapisany.")

model_ts.summary()

# --- Krok 4: Ocena Modelu ---
loss, accuracy = model_ts.evaluate(X_test_scaled, y_test_cat)
print(f"\nDokładność modelu na danych testowych: {accuracy:.3f}")



# --- Krok 5: Praktyczne Użycie Modelu ---
# Weźmy losowy fragment sygnału ze zbioru testowego i zobaczmy, co przewidzi model

# Wybieramy losowy indeks
random_idx = np.random.randint(0, len(X_test_scaled))
sample_signal = X_test_scaled[random_idx]
true_activity_idx = y_test[random_idx]
true_activity_name = CLASS_NAMES[true_activity_idx]

# Model oczekuje "partii" danych, więc dodajemy jeden wymiar na początku
sample_signal_batch = np.expand_dims(sample_signal, axis=0)

# Predykcja
prediction_proba = model_ts.predict(sample_signal_batch)[0]
predicted_activity_idx = np.argmax(prediction_proba)
predicted_activity_name = CLASS_NAMES[predicted_activity_idx]

# Wizualizacja wyniku
print(f"\n--- Test na losowym fragmencie sygnału nr {random_idx} ---")
print(f"Prawdziwa aktywność: {true_activity_name}")
print(f"Przewidziana aktywność: {predicted_activity_name}")
print(f"Pewność predykcji: {np.max(prediction_proba):.2%}")

plt.figure(figsize=(12, 5))
plt.plot(sample_signal)
plt.title(f"Sygnał wejściowy\nPrawda: {true_activity_name} | Predykcja: {predicted_activity_name}")
plt.xlabel("Krok czasowy")
plt.ylabel("Wartość z sensora (skalowana)")
plt.legend(['Oś X', 'Oś Y', 'Oś Z'])
plt.show()