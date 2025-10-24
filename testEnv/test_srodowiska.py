# test_srodowiska.py

# --- KROK 1: Sprawdzenie importów ---
print("--- Sprawdzanie importów bibliotek ---")
try:
    import numpy as np
    print("[OK] NumPy zaimportowany poprawnie.")
except ImportError:
    print("[BŁĄD] Nie udało się zaimportować NumPy.")

try:
    import pandas as pd
    print("[OK] Pandas zaimportowany poprawnie.")
except ImportError:
    print("[BŁĄD] Nie udało się zaimportować Pandas.")

try:
    import matplotlib.pyplot as plt
    print("[OK] Matplotlib zaimportowany poprawnie.")
except ImportError:
    print("[BŁĄD] Nie udało się zaimportować Matplotlib.")

try:
    import sklearn
    print("[OK] Scikit-learn zaimportowany poprawnie.")
except ImportError:
    print("[BŁĄD] Nie udało się zaimportować Scikit-learn.")
    
try:
    import seaborn as sns
    print("[OK] Seaborn zaimportowany poprawnie.")
except ImportError:
    print("[BŁĄD] Nie udało się zaimportować Seaborn.")

print("\n--- Testowanie funkcjonalności ---")

# --- KROK 2: Proste przykłady użycia ---

# NumPy: tworzenie i operacje na tablicach
print("\n[NumPy] Tworzenie tablicy i obliczenia...")
wektor_np = np.array([1, 2, 3, 4, 5])
print(f"  Wektor NumPy: {wektor_np}")
print(f"  Średnia z wektora: {wektor_np.mean()}")
print(f"  Pomnożony przez 2: {wektor_np * 2}")

# Pandas: tworzenie ramki danych (DataFrame)
print("\n[Pandas] Tworzenie ramki danych...")
dane = {'Kolumna A': [10, 20, 30], 'Kolumna B': ['X', 'Y', 'Z']}
df = pd.DataFrame(dane)
print("  Ramka danych (DataFrame):")
print(df)

# Scikit-learn: prosty model
print("\n[Scikit-learn] Test prostego modelu...")
from sklearn.linear_model import LinearRegression
model = LinearRegression()
print(f"  Utworzono model: {type(model).__name__}")
# Przygotowanie prostych danych do regresji: y = 2x
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])
model.fit(X, y)
przewidywanie = model.predict([[4]])
print(f"  Model nauczony. Przewidywanie dla X=4: {przewidywanie[0]:.2f} (oczekiwano ~6.0)")

# Matplotlib i Seaborn: generowanie wykresu
print("\n[Matplotlib & Seaborn] Generowanie wykresu 'test_wykres.png'...")
try:
    # Używamy Seaborn dla lepszego stylu
    sns.set_theme()
    
    x_wykres = np.linspace(0, 10, 100)
    y_wykres = np.sin(x_wykres)
    
    plt.figure(figsize=(8, 5)) # Utworzenie rysunku o określonym rozmiarze
    plt.plot(x_wykres, y_wykres, label='sin(x)')
    plt.title("Testowy wykres funkcji sinus")
    plt.xlabel("Oś X")
    plt.ylabel("Oś Y")
    plt.legend()
    plt.grid(True)
    
    # Zapis wykresu do pliku
    plt.savefig("test_wykres.png")
    print("  Wykres został zapisany do pliku 'test_wykres.png'.")
    plt.show()
    print("  Okienko z wykresem zostało zamknięte.")
except Exception as e:
    print(f"[BŁĄD] Nie udało się wygenerować wykresu: {e}")

print("\n--- Koniec testu środowiska ---")
print("Jeśli nie widzisz żadnych błędów, Twoje środowisko jest gotowe do pracy!")