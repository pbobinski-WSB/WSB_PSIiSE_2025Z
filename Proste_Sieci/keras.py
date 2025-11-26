from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from commons.diabetes_data import X_train_scaled, X_test_scaled, y_train, y_test, pd
from commons.utils import plot_confusion_matrix

# Zakładamy, że X_train_scaled, y_train, X_test_scaled, y_test są już dostępne

print("\n--- Budowa i trening modelu MLP w Keras/TensorFlow ---")

# --- Krok 1: Zdefiniowanie architektury modelu ---
# Tworzymy model sekwencyjny - warstwy będą dodawane jedna po drugiej.
model = Sequential()

# Dodajemy warstwy, używając składni .add()
# Warstwa wejściowa jest definiowana w pierwszej warstwie Dense przez `input_shape`
# Nasze dane mają 8 cech, więc input_shape=(8,)
model.add(Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)))

# Dodajemy drugą warstwę ukrytą
model.add(Dense(16, activation='relu'))

# Dodajemy warstwę wyjściową
# 1 neuron, ponieważ to klasyfikacja binarna (Chory/Zdrowy)
# Aktywacja 'sigmoid', ponieważ chcemy uzyskać prawdopodobieństwo (wartość między 0 a 1)
model.add(Dense(1, activation='sigmoid'))

# --- Krok 2: Podsumowanie i Kompilacja modelu ---
# Wyświetlenie architektury modelu i liczby parametrów do nauczenia
print("\nArchitektura modelu:")
model.summary()

# Kompilacja modelu - konfiguracja procesu uczenia
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# --- Krok 3: Trening modelu ---
# Trenujemy model na danych treningowych
# epochs - ile razy model "zobaczy" cały zbiór danych
# batch_size - ile próbek jest przetwarzanych naraz przed aktualizacją wag
# validation_data - dane do walidacji modelu po każdej epoce
print("\nRozpoczynanie treningu...")
history = model.fit(X_train_scaled, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_test_scaled, y_test),
                    verbose=0) # verbose=0 wyłącza logowanie postępu na ekranie

print("Trening zakończony.")

# --- Krok 4: Ocena modelu ---
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\nDokładność modelu na zbiorze testowym: {accuracy:.3f}")

# --- Krok 5: Predykcja i bardziej szczegółowa analiza ---
# model.predict zwraca prawdopodobieństwa, musimy je zamienić na klasy 0/1
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype("int32")

print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred))

# Wizualizacja macierzy pomyłek
plot_confusion_matrix(y_test, y_pred.flatten(), model_name='Keras MLP')

# --- Krok 6 (Opcjonalny, ale bardzo wartościowy): Wizualizacja historii treningu ---
# Możemy narysować, jak zmieniała się dokładność i funkcja straty w trakcie treningu
history_df = pd.DataFrame(history.history)

plt.figure(figsize=(12, 5))
# Wykres funkcji straty
plt.subplot(1, 2, 1)
plt.plot(history_df['loss'], label='Strata treningowa')
plt.plot(history_df['val_loss'], label='Strata walidacyjna')
plt.title('Historia funkcji straty')
plt.xlabel('Epoka')
plt.ylabel('Strata (Loss)')
plt.legend()

# Wykres dokładności
plt.subplot(1, 2, 2)
plt.plot(history_df['accuracy'], label='Dokładność treningowa')
plt.plot(history_df['val_accuracy'], label='Dokładność walidacyjna')
plt.title('Historia dokładności')
plt.xlabel('Epoka')
plt.ylabel('Dokładność (Accuracy)')
plt.legend()

plt.tight_layout()
plt.show()