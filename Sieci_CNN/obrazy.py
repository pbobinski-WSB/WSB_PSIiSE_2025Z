from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
from commons.cifar10_data import num_classes, x_test, x_train, y_test, y_train, class_names, cifar10, y_test_original
import os

# --- Definicja nazwy pliku modelu ---
model_filename = "cifar10_cnn_model.keras"

# --- GŁÓWNA LOGIKA: WCZYTAJ LUB TRENUJ ---

# Sprawdzamy, czy plik z modelem już istnieje
if os.path.exists(model_filename):
    # --- ŚCIEŻKA 1: Plik istnieje - wczytujemy model ---
    print(f"--- Znaleziono istniejący model '{model_filename}'. Wczytywanie... ---")
    model_cnn = keras.models.load_model(model_filename)
    print("Model został pomyślnie wczytany.")

else:
    # --- ŚCIEŻKA 2: Plik nie istnieje - definiujemy, kompilujemy i trenujemy model ---
    print(f"--- Nie znaleziono modelu '{model_filename}'. Rozpoczynanie nowego treningu... ---")

    # --- Budowa Architektury CNN ---
    model_cnn = Sequential()
    # Pierwsza warstwa konwolucyjna
    model_cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model_cnn.add(MaxPooling2D((2, 2)))
    # Druga warstwa konwolucyjna
    model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
    model_cnn.add(MaxPooling2D((2, 2)))
    # Trzecia warstwa konwolucyjna
    model_cnn.add(Conv2D(64, (3, 3), activation='relu'))

    # Spłaszczenie i dodanie klasyfikatora
    model_cnn.add(Flatten())
    model_cnn.add(Dense(64, activation='relu'))
    model_cnn.add(Dense(num_classes, activation='softmax')) # Softmax dla klasyfikacji wieloklasowej

    
    # --- Kompilacja i Trening ---
    model_cnn.compile(optimizer='adam',
                    loss='categorical_crossentropy', # Używamy dla etykiet one-hot
                    metrics=['accuracy'])

    # Trening może chwilę potrwać, nawet na CPU
    history_cnn = model_cnn.fit(x_train, y_train, epochs=10, 
                                validation_data=(x_test, y_test))

    print("Trening zakończony.")
    plt.plot(history_cnn.history['accuracy'], label='dokładność (trening)')
    plt.plot(history_cnn.history['val_accuracy'], label = 'dokładność (walidacja)')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()
    # 4. Zapis nowo wytrenowanego modelu
    print(f"\nZapisywanie nowego modelu do pliku '{model_filename}'...")
    model_cnn.save(model_filename)
    print("Model pomyślnie zapisany.")

# --- Ocena i Wizualizacja Historii ---
model_cnn.summary()
test_loss, test_acc = model_cnn.evaluate(x_test, y_test, verbose=2)
print(f"\nDokładność na danych testowych: {test_acc:.3f}")


# --- Rozwinięcie: Predykcja na pojedynczych obrazkach ---

# Pobieramy predykcje dla całego zbioru testowego
predictions = model_cnn.predict(x_test)

# Funkcja do wizualizacji pojedynczej predykcji
def plot_prediction(i, predictions_array, true_label, img, class_names):
    true_label, img = true_label[i][0], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    
    true_label_int = int(true_label)
    
    # Kolorujemy tytuł na zielono dla poprawnej predykcji, na czerwono dla błędnej
    if predicted_label == true_label_int:
        color = 'green'
    else:
        color = 'red'
    
    plt.xlabel(f"Predykcja: {class_names[predicted_label]} ({100*np.max(predictions_array):.2f}%)\n"
               f"Prawda: {class_names[true_label_int]}", # Używamy nowej zmiennej
               color=color)


# Wyświetlmy kilka losowych obrazków z ich predykcjami
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    # Wybieramy losowy indeks
    idx = np.random.randint(0, x_test.shape[0])
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_prediction(idx, predictions[idx], y_test_original, x_test, class_names) # y_test_original to y_test przed one-hot encoding
plt.tight_layout()
plt.show()

# y_test_original to (test_labels, ), czyli y_test przed to_categorical
# Jeśli go nie masz, wczytaj dane jeszcze raz: (_, _), (test_images, test_labels) = cifar10.load_data()
(_, _), (_, y_test_original) = cifar10.load_data()