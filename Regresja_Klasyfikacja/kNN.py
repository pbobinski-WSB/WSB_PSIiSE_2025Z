# k Najbliższych sąsiadów - Klasyfikacja: Przewidywanie, czy pacjent ma cukrzycę. 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from commons.diabetes_data import X_train_scaled, y_train, X_test_scaled, y_test
from commons.utils import plot_confusion_matrix

# 1. Inicjalizacja i trening modelu k-NN (wybierzmy np. k=7)
model_knn = KNeighborsClassifier(n_neighbors=7)
model_knn.fit(X_train_scaled, y_train)

# 2. Predykcja i ocena
y_pred_knn = model_knn.predict(X_test_scaled)

print("\n\n--- Wyniki k-Najbliższych Sąsiadów (k-NN) ---")
print(f"Dokładność: {accuracy_score(y_test, y_pred_knn):.2f}")
print("\nMacierz Pomyłek:")
print(confusion_matrix(y_test, y_pred_knn))
print("\nRaport Klasyfikacji:")
print(classification_report(y_test, y_pred_knn))

plot_confusion_matrix(y_test, y_pred_knn,'k-NN')

