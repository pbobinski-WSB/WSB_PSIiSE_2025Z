# --- Przykład 2: Generowanie Tekstu ---
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
import os

# --- Krok 1: Przygotowanie danych ---
# Użyjemy prostego, znanego wierszyka jako naszego korpusu do nauki
text = "mary had a little lamb little lamb little lamb mary had a little lamb whose fleece was white as snow"

# Stworzenie "słownika" unikalnych znaków
chars = sorted(list(set(text)))
char_to_int = {char: i for i, char in enumerate(chars)}
int_to_char = {i: char for i, char in enumerate(chars)}

n_chars = len(text)
n_vocab = len(chars)
print("Liczba wszystkich znaków w tekście:", n_chars)
print("Liczba unikalnych znaków (rozmiar słownika):", n_vocab)

# Przygotowanie zbioru danych: sekwencje wejściowe (X) i znak wyjściowy (y)
seq_length = 10  # Długość sekwencji wejściowej
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print("Liczba sekwencji treningowych:", n_patterns)

# Przygotowanie danych wejściowych do formatu oczekiwanego przez LSTM
# Kształt: [próbki, kroki_czasowe, cechy]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# Normalizacja
X = X / float(n_vocab)
# Konwersja etykiet wyjściowych na one-hot encoding
y = to_categorical(dataY)


# --- Definicja nazwy pliku modelu ---
model_filename = "gen_lstm_model.keras"

# --- GŁÓWNA LOGIKA: WCZYTAJ LUB TRENUJ ---

# Sprawdzamy, czy plik z modelem już istnieje
if os.path.exists(model_filename):
    # --- ŚCIEŻKA 1: Plik istnieje - wczytujemy model ---
    print(f"--- Znaleziono istniejący model '{model_filename}'. Wczytywanie... ---")
    model_gen = keras.models.load_model(model_filename)
    print("Model został pomyślnie wczytany.")

else:
    # --- ŚCIEŻKA 2: Plik nie istnieje - definiujemy, kompilujemy i trenujemy model ---
    print(f"--- Nie znaleziono modelu '{model_filename}'. Rozpoczynanie nowego treningu... ---")


    # --- Krok 2: Budowa modelu LSTM ---
    model_gen = Sequential([
        # Używamy prostego LSTM, a nie Embedding, bo wejście jest już znormalizowane
        LSTM(256, input_shape=(X.shape[1], X.shape[2])),
        Dense(n_vocab, activation='softmax')
    ])

    # Kompilacja modelu
    model_gen.compile(loss='categorical_crossentropy', optimizer='adam')

    # --- Krok 3: Trening modelu ---
    # Trening może zająć chwilę
    model_gen.fit(X, y, epochs=100, batch_size=64, verbose=2)

    print(f"\nZapisywanie nowego modelu do pliku '{model_filename}'...")
    model_gen.save(model_filename)
    print("Model pomyślnie zapisany.")

# --- Krok 4: Generowanie nowego tekstu ---
print("\n--- Rozpoczynanie generowania tekstu ---")

# Wybieramy losowy "seed" (ziarno) z naszego tekstu jako początek
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed (ziarno):", "\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# Generujemy 100 znaków
generated_text = ""
for i in range(100):
    # Przygotowujemy wejście do modelu
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    
    # Predykcja następnego znaku
    prediction = model_gen.predict(x, verbose=0)
    
    # Wybieramy znak o najwyższym prawdopodobieństwie
    index = np.argmax(prediction)
    result = int_to_char[index]
    
    # Dodajemy wygenerowany znak do tekstu
    generated_text += result
    
    # Aktualizujemy "ziarno" - usuwamy pierwszy znak i dodajemy nowy na końcu
    pattern.append(index)
    pattern = pattern[1:]

print("\nWygenerowany tekst:")
print(generated_text)