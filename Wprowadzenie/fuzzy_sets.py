# --- Przykład 2: Logika Rozmyta ---

# Import potrzebnych bibliotek
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Krok 1: Zdefiniowanie Zmiennych Lingwistycznych (wejścia i wyjście)
# Antecedents - przyczyny (wejścia)
# Consequent - skutek (wyjście)

# Definiujemy uniwersum dyskusji (np. oceny w skali 0-10, napiwek 0-25%)
jakość = ctrl.Antecedent(np.arange(0, 11, 1), 'jakość') # Jakość jedzenia
obsługa = ctrl.Antecedent(np.arange(0, 11, 1), 'obsługa') # Jakość obsługi
napiwek = ctrl.Consequent(np.arange(0, 26, 1), 'napiwek') # Wysokość napiwku

# Krok 2: Zdefiniowanie Zbiorów Rozmytych (np. słaba, dobra, wspaniała)
# Używamy funkcji trójkątnych i trapezoidalnych do zdefiniowania przynależności
jakość.automf(names=['słaba', 'dobra', 'wspaniała'])
obsługa.automf(names=['słaba', 'dobra', 'wspaniała'])

napiwek['niski'] = fuzz.trimf(napiwek.universe, [0, 0, 13])
napiwek['średni'] = fuzz.trimf(napiwek.universe, [0, 13, 25])
napiwek['wysoki'] = fuzz.trimf(napiwek.universe, [13, 25, 25])

# Możemy zwizualizować te zbiory, żeby zobaczyć jak wyglądają
# obsługa.view()
# napiwek.view()

# Krok 3: Zdefiniowanie Reguł Eksperckich
# To jest "baza wiedzy" naszego systemu
reguła1 = ctrl.Rule(obsługa['słaba'] | jakość['słaba'], napiwek['niski'])
reguła2 = ctrl.Rule(obsługa['dobra'], napiwek['średni'])
reguła3 = ctrl.Rule(obsługa['wspaniała'] | jakość['wspaniała'], napiwek['wysoki'])

# Krok 4: Zbudowanie i symulacja systemu sterowania
system_sterowania = ctrl.ControlSystem([reguła1, reguła2, reguła3])
symulacja_napiwku = ctrl.ControlSystemSimulation(system_sterowania)

# Krok 5: Uruchomienie symulacji z konkretnymi danymi
# Załóżmy, że oceniamy obsługę na 9.8/10, a jedzenie na 6.5/10
symulacja_napiwku.input['obsługa'] = 9.8
symulacja_napiwku.input['jakość'] = 6.5

# Obliczenie wyniku
symulacja_napiwku.compute()

# Wyświetlenie wyniku (po "wyostrzeniu" - defuzzyfikacji)
print(f"Sugerowana wysokość napiwku: {symulacja_napiwku.output['napiwek']:.2f}%")

# Wyświetlenie wykresu wynikowego
# napiwek.view(sim=symulacja_napiwku)