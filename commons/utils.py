import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def parse_experiment_ranges(range_str):
    """Parsuje string z zakresami (np. "1-3,5,7") na zbiór numerów eksperymentów."""
    selected = set()
    try:
        parts = range_str.split(',')
        for part in parts:
            if '-' in part:
                start, end = map(int, part.split('-'))
                selected.update(range(start, end + 1))
            else:
                selected.add(int(part))
    except ValueError:
        print(f"BŁĄD: Nieprawidłowy format argumentu --experiments. Użyj formatu '1-3,5,7'.")
        sys.exit(1) # Wyjście z programu z kodem błędu
    return selected

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Generuje i wyświetla wizualizację macierzy pomyłek dla danego modelu.

    Args:
        y_true (array-like): Rzeczywiste etykiety (np. y_test).
        y_pred (array-like): Etykiety przewidziane przez model.
        model_name (str): Nazwa modelu, która zostanie użyta w tytule wykresu 
                          (np. "Regresja Logistyczna").
    """
    # Obliczenie macierzy pomyłek
    cm = confusion_matrix(y_true, y_pred)

    # Ustawienie kolorów w zależności od nazwy modelu dla estetyki
    if "Regresja" in model_name:
        cmap = "Blues"
    elif "k-NN" in model_name:
        cmap = "Greens"
    else:
        cmap = "viridis"

    # Stworzenie figury dla pojedynczego wykresu
    plt.figure(figsize=(8, 6))

    # Rysowanie mapy ciepła (heatmap)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=['Zdrowy (0)', 'Chory (1)'], 
                yticklabels=['Zdrowy (0)', 'Chory (1)'])

    # Ustawienie tytułu i etykiet
    plt.title(f'Macierz Pomyłek - {model_name}')
    plt.xlabel('Etykieta Przewidziana')
    plt.ylabel('Etykieta Rzeczywista')

    # Wyświetlenie wykresu
    plt.show()

# Funkcja pomocnicza do rysowania granic decyzyjnych
def plot_decision_boundary(model, X, y, title):
    # Skalowanie siatki
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predykcja na każdym punkcie siatki
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Rysowanie wykresu
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
    plt.title(title)
    plt.xlabel("Cecha 1")
    plt.ylabel("Cecha 2")
    plt.show()

import tensorflow as tf
import tensorflow_datasets as tfds
import sounddevice as sd
    
def play_audio_from_tfds_index(idx):
    ds_test_original, ds_info = tfds.load('speech_commands', split='test', shuffle_files=False, with_info=True)
    
    audio_data_to_play = None
    for element in ds_test_original:
        if element['label'].numpy() == idx:
            audio_data = element['audio']
            sample_rate = ds_info.features['audio'].sample_rate
            
            # Normalizujemy tensor do zakresu float [-1.0, 1.0], którego oczekuje sounddevice
            audio_data_to_play = tf.cast(audio_data, tf.float32) / 32768.0
            break
    
    if audio_data_to_play is not None:
        print("\nOdtwarzanie oryginalnego pliku audio (potrwa ok. 1 sekundę)...")
        try:
            # Używamy sounddevice.play() do odtworzenia sygnału
            # `blocking=True` sprawia, że skrypt poczeka, aż dźwięk się skończy
            sd.play(audio_data_to_play.numpy(), samplerate=sample_rate, blocking=True)
            print("Odtwarzanie zakończone.")
        except Exception as e:
            print(f"Nie udało się odtworzyć dźwięku. Błąd: {e}")
            print("Upewnij się, że masz poprawnie skonfigurowane urządzenia audio.")

