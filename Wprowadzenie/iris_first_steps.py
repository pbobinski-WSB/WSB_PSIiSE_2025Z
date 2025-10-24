# 1. Importowanie potrzebnych bibliotek
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Załadowanie zbioru danych
iris_dataset = load_iris()
X = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)
y = pd.Series(iris_dataset.target)

# Szybki podgląd danych
print("Pierwsze 5 wierszy danych (cechy):")
print(X.head())
print("\nEtykiety (0=Setosa, 1=Versicolor, 2=Virginica):")
print(y.head())

# Użyjemy biblioteki seaborn do stworzenia wykresu par (pair plot)
# Kolorujemy punkty według gatunku irysa (y)
df_to_plot = X.copy()
df_to_plot['species'] = y.map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

sns.pairplot(df_to_plot, hue='species')
plt.show()

# 3. Podział danych na zbiór treningowy (do nauki) i testowy (do oceny)
# test_size=0.3 oznacza, że 30% danych trafi do zbioru testowego
# random_state=42 zapewnia powtarzalność podziału
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Wybór i inicjalizacja modelu
# Użyjemy prostego klasyfikatora k-Najbliższych Sąsiadów (k-NN) z k=3
model = KNeighborsClassifier(n_neighbors=3)

# 5. Trening modelu na danych treningowych
model.fit(X_train, y_train)

# 6. Przewidywanie etykiet dla danych testowych (których model nie widział)
y_pred = model.predict(X_test)

# 7. Ocena jakości modelu - jak dobrze sobie poradził?
accuracy = accuracy_score(y_test, y_pred)
print(f"\nDokładność naszego pierwszego modelu: {accuracy:.2f}") 
# Oczekiwany wynik: 1.00 (dla tego podziału i modelu)