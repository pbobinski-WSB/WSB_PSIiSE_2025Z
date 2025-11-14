import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs # Do generowania syntetycznych danych

# --- Krok 1: Generowanie i wizualizacja danych ---
# Stworzymy syntetyczny zbiór danych przypominający dane o klientach
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
df = pd.DataFrame(X, columns=['Roczny Dochód (skalowany)', 'Wskaźnik Wydatków (skalowany)'])

print("Pierwsze 5 wierszy danych:")
print(df.head())

# Wizualizacja "surowych" danych
sns.scatterplot(x='Roczny Dochód (skalowany)', y='Wskaźnik Wydatków (skalowany)', data=df)
plt.title('Dane klientów przed klastrowaniem')
plt.show()

# --- Krok 2: Metoda Łokcia do znalezienia optymalnego K ---
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

# Rysowanie wykresu Metody Łokcia
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.xlabel('Liczba klastrów (K)')
plt.ylabel('WCSS (Inertia)')
plt.title('Metoda Łokcia')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# --- Krok 3: Budowa finalnego modelu k-Means z optymalnym K ---
# Zgodnie z Metodą Łokcia, wybieramy K=4
final_kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Klaster'] = final_kmeans.fit_predict(df[['Roczny Dochód (skalowany)', 'Wskaźnik Wydatków (skalowany)']])

# Pobranie współrzędnych centroid
centroids = final_kmeans.cluster_centers_

print("\nWyniki klastrowania (pierwsze 5 wierszy):")
print(df.head())

# --- Krok 4: Wizualizacja wyników klastrowania ---
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Roczny Dochód (skalowany)', y='Wskaźnik Wydatków (skalowany)', 
                hue='Klaster', data=df, palette='viridis', s=100)
# Narysowanie centroid
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroidy')
plt.title('Segmentacja Klientów za pomocą k-Means (K=4)')
plt.legend()
plt.show()