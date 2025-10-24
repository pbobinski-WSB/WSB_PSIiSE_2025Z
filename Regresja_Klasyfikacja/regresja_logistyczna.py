# Regresja logistyczna - Klasyfikacja: Przewidywanie, czy pacjent ma cukrzycę. 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from commons.diabetes_data import X_train_scaled, y_train, X_test_scaled, y_test
from sklearn.metrics import confusion_matrix
from commons.utils import plot_confusion_matrix

# 4. Inicjalizacja i trening modelu
model_log = LogisticRegression(random_state=42)
model_log.fit(X_train_scaled, y_train)

# 5. Predykcja i ocena
y_pred_log = model_log.predict(X_test_scaled)

print("--- Wyniki Regresji Logistycznej ---")
print(f"Dokładność: {accuracy_score(y_test, y_pred_log):.2f}")
print("\nMacierz Pomyłek:")
print(confusion_matrix(y_test, y_pred_log))
print("\nRaport Klasyfikacji:")
print(classification_report(y_test, y_pred_log))

plot_confusion_matrix(y_test, y_pred_log,'Regresja')
