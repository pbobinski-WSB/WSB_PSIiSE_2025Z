# Regresja: Przewidywanie cen domów w Kalifornii.

# 1. Importy i ładowanie danych
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# 2. Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Inicjalizacja i trening modelu
model_reg = LinearRegression()
model_reg.fit(X_train, y_train)

# 4. Predykcja na danych testowych
y_pred_reg = model_reg.predict(X_test)

# 5. Ocena modelu
mse = mean_squared_error(y_test, y_pred_reg)
r2 = r2_score(y_test, y_pred_reg)
print(f"Błąd średnio-kwadratowy (MSE): {mse:.2f}")
print(f"Współczynnik determinacji (R^2): {r2:.2f}") # Jaką część wariancji wyjaśnia model

# 6. Wizualizacja
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_reg, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2)
plt.xlabel("Wartości Rzeczywiste")
plt.ylabel("Wartości Przewidziane")
plt.title("Rzeczywiste vs Przewidziane wartości")
plt.show()

# INTERPRETACJA WYNIKÓW
# Analizujemy współczynniki (wagi), których nauczył się model.
# Mówią nam one, jak zmiana o jednostkę w danej cesze wpływa na cenę.

coefficients = pd.DataFrame(model_reg.coef_, X.columns, columns=['Współczynnik'])
coefficients = coefficients.sort_values(by='Współczynnik', ascending=False)

print("\n--- Wpływ poszczególnych cech na cenę domu ---")
print(coefficients)

# Wizualizacja współczynników
plt.figure(figsize=(10, 8))
sns.barplot(x=coefficients.index, y=coefficients['Współczynnik'], palette='viridis', hue=coefficients.index, legend=False)
plt.title('Współczynniki modelu regresji liniowej')
plt.xlabel('Cecha')
plt.ylabel('Wpływ na cenę (w $100k)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

