import pandas as pd
from sklearn.metrics import accuracy_score
from commons.diabetes_data import X_train, y_train, X_test, y_test, X, y
from commons.utils import plt, sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

X_diab, y_diab = X, y

# Podział na zbiór treningowy i testowy
X_train_diab, X_test_diab, y_train_diab, y_test_diab = X_train, X_test, y_train, y_test

# --- Ważna uwaga: Skalowanie Danych ---
# W przeciwieństwie do Modułu 2, dla modeli drzewiastych (Drzewa Decyzyjne, Lasy Losowe)
# skalowanie cech NIE JEST konieczne. 
# Modele te podejmują decyzje na podstawie progów (np. "czy Glukoza > 127.5?"), a skalowanie nie zmienia wyniku takiego porównania.
# Dlatego pracujemy na oryginalnych, nieskalowanych danych, co ułatwia interpretację.


# --- Krok 2: Wizualizacja małego Drzewa Decyzyjnego ---
# Budujemy małe, proste drzewo (o maksymalnej głębokości 3), aby móc je zwizualizować
# i zrozumieć jego logikę decyzyjną.

print("--- Budowa i wizualizacja prostego Drzewa Decyzyjnego ---")
small_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
small_tree.fit(X_train_diab, y_train_diab)

# Rysowanie struktury drzewa
plt.figure(figsize=(22, 12)) # Ustawiamy duży rozmiar, aby wykres był czytelny
plot_tree(small_tree, 
          feature_names=X_diab.columns, 
          class_names=['Zdrowy', 'Chory'], 
          filled=True, 
          rounded=True,
          fontsize=12)
plt.title("Wizualizacja Drzewa Decyzyjnego (max_depth=3) dla danych o cukrzycy", fontsize=16)
plt.show()


# --- Krok 3: Budowa Lasu Losowego i ocena jego skuteczności ---
# Teraz budujemy potężniejszy model - Las Losowy, składający się ze 100 drzew.

print("\n--- Budowa i ocena Lasu Losowego ---")
forest = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
forest.fit(X_train_diab, y_train_diab)

# Predykcja i ocena na zbiorze testowym
y_pred_forest = forest.predict(X_test_diab)
accuracy = accuracy_score(y_test_diab, y_pred_forest)
print(f"Dokładność Lasu Losowego na zbiorze testowym: {accuracy:.3f}")


# --- Krok 4: Analiza Ważności Cech (Feature Importance) ---
# Wykorzystujemy Las Losowy, aby dowiedzieć się, które cechy miały największy
# wpływ na decyzje podejmowane przez model.

print("\n--- Analiza Ważności Cech ---")
# Pobranie ważności cech z wytrenowanego modelu
importances = forest.feature_importances_

# Stworzenie DataFrame dla łatwiejszej wizualizacji
feature_df = pd.DataFrame({
    'Cecha': X_diab.columns,
    'Ważność': importances
})

# Posortowanie cech od najważniejszej do najmniej ważnej
feature_df = feature_df.sort_values(by='Ważność', ascending=False)

# Wizualizacja ważności cech na wykresie słupkowym
plt.figure(figsize=(12, 8))
sns.barplot(x='Ważność', y='Cecha', data=feature_df, palette='viridis', hue='Cecha', legend=False)
plt.title('Ważność cech według Lasu Losowego dla problemu cukrzycy', fontsize=16)
plt.xlabel('Ważność', fontsize=12)
plt.ylabel('Cecha', fontsize=12)
plt.show()

# Wyświetlenie rankingu w formie tabeli
print("\nRanking najważniejszych cech:")
print(feature_df)