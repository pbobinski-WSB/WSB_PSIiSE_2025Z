from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from commons.diabetes_data import X_train_scaled, X_test_scaled, y_train, y_test, pd
from commons.utils import plot_confusion_matrix

# --- Importy dodatkowych narzędzi ---
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Krok 1 (ZMODYFIKOWANY): Zdefiniowanie architektury z warstwami Dropout ---
print("\n--- Budowa i trening DOSTROJONEGO modelu MLP w Keras/TensorFlow ---")
model_tuned = Sequential()

model_tuned.add(Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)))
# Dodajemy warstwę Dropout po pierwszej warstwie Dense.
# "Wyłączy" ona losowo 30% neuronów podczas treningu.
model_tuned.add(Dropout(0.3))

model_tuned.add(Dense(16, activation='relu'))
# Dodajemy kolejną warstwę Dropout.
model_tuned.add(Dropout(0.3))

model_tuned.add(Dense(1, activation='sigmoid'))

# --- Krok 2: Podsumowanie i Kompilacja (bez zmian) ---
print("\nArchitektura dostrojonego modelu:")
model_tuned.summary()

model_tuned.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

# --- Krok 3 (ZMODYFIKOWANY): Trening z Early Stopping ---
# Definiujemy callback EarlyStopping
# monitor='val_loss' -> śledzimy stratę na zbiorze walidacyjnym
# patience=15 -> przerwij trening, jeśli strata nie poprawi się przez 15 kolejnych epok
# restore_best_weights=True -> przywróć najlepsze wagi znalezione podczas treningu
early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)

print("\nRozpoczynanie treningu z Early Stopping i Dropout...")
# Zwiększamy liczbę epok do dużej wartości (np. 200), bo EarlyStopping i tak przerwie trening w optymalnym momencie.
history_tuned = model_tuned.fit(X_train_scaled, y_train,
                                epochs=200,
                                batch_size=32,
                                validation_data=(X_test_scaled, y_test),
                                # Dodajemy nasz callback do listy
                                callbacks=[early_stop],
                                verbose=0)

print("Trening zakończony (prawdopodobnie przez Early Stopping).")

# --- Krok 4: Ocena dostrojonego modelu ---
loss_tuned, accuracy_tuned = model_tuned.evaluate(X_test_scaled, y_test)
print(f"\nDokładność DOSTROJONEGO modelu na zbiorze testowym: {accuracy_tuned:.3f}")

# --- Krok 5: Predykcja i analiza ---
y_pred_tuned_proba = model_tuned.predict(X_test_scaled)
y_pred_tuned = (y_pred_tuned_proba > 0.5).astype("int32")

print("\nRaport klasyfikacji (dostrojony model):")
print(classification_report(y_test, y_pred_tuned))

plot_confusion_matrix(y_test, y_pred_tuned.flatten(), model_name='Keras MLP (Dostrojony)')

# --- Krok 6: Wizualizacja nowej historii treningu ---
history_df_tuned = pd.DataFrame(history_tuned.history)
plt.figure(figsize=(12, 5))
# Wykres funkcji straty
plt.subplot(1, 2, 1)
plt.plot(history_df_tuned['loss'], label='Strata treningowa')
plt.plot(history_df_tuned['val_loss'], label='Strata walidacyjna')
plt.title('Historia funkcji straty (Dostrojony Model)')
plt.xlabel('Epoka')
plt.ylabel('Strata (Loss)')
plt.legend()

# Wykres dokładności
plt.subplot(1, 2, 2)
plt.plot(history_df_tuned['accuracy'], label='Dokładność treningowa')
plt.plot(history_df_tuned['val_accuracy'], label='Dokładność walidacyjna')
plt.title('Historia dokładności (Dostrojony Model)')
plt.xlabel('Epoka')
plt.ylabel('Dokładność (Accuracy)')
plt.legend()

plt.tight_layout()
plt.show()