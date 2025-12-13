# --- Import niezbędnych bibliotek ---
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow_datasets as tfds  # <-- NOWY, WAŻNY IMPORT
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Resizing

from commons.utils import play_audio_from_tfds_index

# --- Ustawienie estetyki wykresów ---
sns.set_theme(style="whitegrid")

# --- Krok 1: Pobranie i przygotowanie danych z tensorflow_datasets ---

# Wczytujemy zbiór danych speech_commands. 
# split=['train', 'validation', 'test'] pobiera wszystkie dostępne podzbiory
# with_info=True daje nam dodatkowe informacje o zbiorze (np. nazwy klas)
(ds_train, ds_validation, ds_test), ds_info = tfds.load(
    'speech_commands',
    split=['train', 'validation', 'test'],
    shuffle_files=True,
    with_info=True,
)

# Pobranie nazw komend
commands = ds_info.features['label'].names
print('Komendy:', commands)
NUM_CLASSES = len(commands)

# Funkcja do przekształcania audio na spektrogram
def audio_to_spectrogram(element):
    """
    Przekształca jeden element ze zbioru danych (słownik) na spektrogram i etykietę.
    """
    # 1. Rozpakowujemy słownik
    audio_tensor = element['audio']
    label = element['label']
    
    # To jest standardowa długość dla tego zbioru danych.
    target_len = 16000
    audio_len = tf.shape(audio_tensor)[0]
    
    # Dopełnij zerami, jeśli sygnał jest za krótki
    if audio_len < target_len:
        padding = tf.zeros([target_len - audio_len], dtype=tf.int16)
        audio_tensor = tf.concat([audio_tensor, padding], 0)
    # Przytnij, jeśli sygnał jest za długi
    elif audio_len > target_len:
        audio_tensor = audio_tensor[:target_len]
    
    # Normalizujemy sygnał do float
    audio_float = tf.cast(audio_tensor, tf.float32) / 32768.0

    
    # Utworzenie spektrogramu
    spectrogram = tf.signal.stft(audio_float, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1) # Dodanie wymiaru kanału
    
    return spectrogram, label

# Mapujemy naszą funkcję na każdy element w zbiorze danych
# .map() to bardzo wydajny sposób na przetwarzanie danych w TF
ds_train = ds_train.map(audio_to_spectrogram)
ds_validation = ds_validation.map(audio_to_spectrogram)
# ds_test = ds_test.map(ds_test)
ds_test = ds_test.map(audio_to_spectrogram)
# Optymalizacja potoku danych
ds_train = ds_train.batch(32).prefetch(tf.data.AUTOTUNE)
ds_validation = ds_validation.batch(32).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.batch(32).prefetch(tf.data.AUTOTUNE)


# --- Wizualizacja przykładowego spektrogramu ---
for example_spectrogram, example_label in ds_train.take(1):
    plt.figure(figsize=(10, 4))
    plt.imshow(tf.math.log(example_spectrogram[0, :, :, 0] + 1e-6).numpy().T, aspect='auto', origin='lower')
    plt.title(f'Spektrogram dla komendy: "{commands[example_label[0]]}"')
    plt.xlabel('Czas')
    plt.ylabel('Częstotliwość')
    plt.show()
    INPUT_SHAPE = example_spectrogram.shape[1:]

# --- Krok 2: Logika "Wczytaj lub Trenuj" ---
model_filename = "audio_digit_classifier_tfds.keras"

if os.path.exists(model_filename):
    print(f"\n--- Znaleziono istniejący model '{model_filename}'. Wczytywanie... ---")
    model_audio = keras.models.load_model(model_filename)
else:
    print(f"\n--- Nie znaleziono modelu. Rozpoczynanie nowego treningu... ---")
    
    # Budowa modelu CNN 2D
    model_audio = Sequential([
        Input(shape=INPUT_SHAPE),
        Resizing(32, 32), # Ujednolicamy rozmiar spektrogramów
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model_audio.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy', # <-- WAŻNA ZMIANA
                      metrics=['accuracy'])
                      
    history = model_audio.fit(ds_train,
                              epochs=10,
                              validation_data=ds_validation)
                              
    model_audio.save(model_filename)
    print(f"Model pomyślnie zapisany w pliku: {model_filename}")

# --- Krok 3: Ocena i Praktyczne Użycie ---
print("\n--- Ocena finalnego modelu ---")
model_audio.summary()
loss, accuracy = model_audio.evaluate(ds_test, verbose=0)
print(f"Dokładność modelu na danych testowych: {accuracy:.3f}")

# Wybierzmy losowy przykład ze zbioru testowego
for test_spectrogram, test_label in ds_test.take(1):
    sample_spectrogram_batch = test_spectrogram[:1] # Bierzemy pierwszy element z partii
    true_label_idx = test_label[0].numpy()

    prediction_proba = model_audio.predict(sample_spectrogram_batch)[0]
    predicted_label_idx = np.argmax(prediction_proba)

    print(f"\n--- Test na losowym pliku audio ---")
    print(f"Prawdziwa komenda: {commands[true_label_idx]}")
    print(f"Przewidziana komenda: {commands[predicted_label_idx]}")

    play_audio_from_tfds_index(true_label_idx)
    play_audio_from_tfds_index(predicted_label_idx)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(tf.math.log(sample_spectrogram_batch[0, :, :, 0] + 1e-6).numpy().T, aspect='auto', origin='lower')
    plt.title("Spektrogram wejściowy")

    plt.subplot(1, 2, 2)
    sns.barplot(x=prediction_proba, y=commands, orient='h')
    plt.title("Prawdopodobieństwa predykcji")
    plt.xlabel("Prawdopodobieństwo")
    plt.tight_layout()
    plt.show()