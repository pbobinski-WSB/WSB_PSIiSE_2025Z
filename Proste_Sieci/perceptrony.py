from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from commons.diabetes_data import X_train_scaled, X_test_scaled, y_train, y_test
from commons.utils import plot_confusion_matrix

print("Dane przygotowane i przeskalowane. Gotowe do modelowania.")
print(f"Rozmiar zbioru treningowego: {X_train_scaled.shape}")
print(f"Rozmiar zbioru testowego: {X_test_scaled.shape}")

# --- Inicjalizacja i trening modelu Perceptron ---
print("\n--- Perceptron ---")
perceptron = Perceptron(random_state=42, max_iter=1000, tol=1e-3)
perceptron.fit(X_train_scaled, y_train)

# --- Predykcja i ocena ---
y_pred_perceptron = perceptron.predict(X_test_scaled)

print(f"Dokładność (Accuracy): {accuracy_score(y_test, y_pred_perceptron):.3f}")
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred_perceptron))

# Wizualizacja macierzy pomyłek
plot_confusion_matrix(y_test, y_pred_perceptron, model_name='Perceptron')

# --- Inicjalizacja i trening modelu MLP ---
print("\n--- Perceptron Wielowarstwowy (MLP) ---")

# Definiujemy architekturę: dwie warstwy ukryte, z 50 i 25 neuronami.
mlp = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42,
                    activation='relu', solver='adam', learning_rate_init=0.001)
mlp.fit(X_train_scaled, y_train)

# --- Predykcja i ocena ---
y_pred_mlp = mlp.predict(X_test_scaled)

print(f"Dokładność (Accuracy): {accuracy_score(y_test, y_pred_mlp):.3f}")
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred_mlp))

# Wizualizacja macierzy pomyłek
plot_confusion_matrix(y_test, y_pred_mlp, model_name='MLP')

print("--- Eksperyment 2 (Wersja Poprawiona): MLP ---")

# Zwiększamy max_iter, aby dać modelowi szansę na zbieżność
# Spróbujmy też nieco prostszej architektury, np. jedna warstwa
mlp_tuned = MLPClassifier(hidden_layer_sizes=(100,), # Jedna warstwa ze 100 neuronami
                          max_iter=1500,           # Znacznie więcej iteracji
                          random_state=42,
                          activation='relu',
                          solver='adam')
mlp_tuned.fit(X_train_scaled, y_train)

# Predykcja i ocena
y_pred_mlp_tuned = mlp_tuned.predict(X_test_scaled)

print("\nRaport klasyfikacji - MLP (dostrojony):")

print(f"Dokładność (Accuracy): {accuracy_score(y_test, y_pred_mlp_tuned):.3f}")
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred_mlp_tuned))

# Wizualizacja macierzy pomyłek
plot_confusion_matrix(y_test, y_pred_mlp_tuned, model_name='MLP tuned')