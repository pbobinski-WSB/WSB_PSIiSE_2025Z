import yfinance as yf

import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Ustawienie estetyki wykresów
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (15, 7) # Ustawienie domyślnego rozmiaru wykresów

# --- 1. Stworzenie obiektu Ticker dla Apple Inc. ---
aapl_ticker = yf.Ticker("AAPL")

# --- 2. Pobranie maksymalnej ilości danych historycznych ---
hist_data = aapl_ticker.history(period="max")

# --- 3. Wizualizacja nr 1: Historyczna cena zamknięcia ---
print("Wizualizacja historycznej ceny akcji Apple...")
plt.plot(hist_data['Close'], label='Cena Zamknięcia (Close)')
plt.title('Historyczna cena akcji Apple (AAPL)')
plt.xlabel('Data')
plt.ylabel('Cena (USD)')
plt.legend()
plt.show()

# --- Wizualizacja nr 2: Wolumen obrotu (WERSJA OSTATECZNA) ---
print("\nWizualizacja wolumenu obrotu...")

plt.figure(figsize=(15, 7)) # Upewnijmy się, że ten wykres też ma duży rozmiar

# ==============================================================================
# ZMIANA Z plt.bar NA plt.fill_between
# Rysujemy wypełniony obszar pod linią danych wolumenu.
# alpha=0.5 nadaje mu lekką przezroczystość, co wygląda profesjonalnie.
# ==============================================================================
plt.fill_between(hist_data.index, hist_data['Volume'], color='steelblue', alpha=0.5, label='Wolumen')

# Opcjonalnie: Możemy dodać cienką linię na krawędzi dla lepszego kontrastu
plt.plot(hist_data.index, hist_data['Volume'], color='darkblue', linewidth=0.5)

# Tytuł i etykiety (bez zmian)
plt.title('Wolumen obrotu akcjami Apple (AAPL)')
plt.xlabel('Data')
plt.ylabel('Liczba akcji')
plt.legend()

# Formatowanie osi Y (bez zmian, wciąż bardzo przydatne)
def millions_formatter(x, pos):
    return f'{x / 1_000_000:.0f}M'

formatter = FuncFormatter(millions_formatter)
plt.gca().yaxis.set_major_formatter(formatter)

# Ustawienie limitów osi, aby usunąć pustą przestrzeń na górze
plt.ylim(0, hist_data['Volume'].max() * 1.1)

plt.tight_layout()
plt.show()

# --- 5. Wizualizacja nr 3: Dywidendy ---
# Dywidendy to część zysku, którą spółka wypłaca akcjonariuszom.
dividends = aapl_ticker.dividends
if not dividends.empty:
    print("\nWizualizacja wypłaconych dywidend...")
    plt.stem(dividends.index, dividends.values, basefmt=" ")
    plt.title('Historia wypłat dywidend przez Apple (AAPL)')
    plt.xlabel('Data')
    plt.ylabel('Dywidenda na akcję (USD)')
    plt.show()
else:
    print("\nSpółka nie wypłacała dywidend w danym okresie.")


import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Ustawienie estetyki wykresów
sns.set_theme(style="darkgrid")
plt.rcParams['figure.figsize'] = (15, 8) # Ustawienie domyślnego rozmiaru wykresów

# --- 1. Pobranie danych dla kilku spółek technologicznych od początku 2020 roku ---
# Używamy listy symboli. yfinance zwróci DataFrame z wielopoziomowymi kolumnami.
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
data = yf.download(tickers, start='2020-01-01')

print("Pobrane dane (fragment):")
# Wyświetlamy tylko ceny zamknięcia, żeby było czytelniej
print(data['Close'].head())

# --- 2. Wizualizacja nr 1: Surowe ceny zamknięcia ---
# Ten wykres pokazuje, jak zmieniały się ceny nominalne.
print("\nWizualizacja surowych cen zamknięcia...")
data['Close'].plot()
plt.title('Historyczne ceny zamknięcia dla spółek technologicznych')
plt.xlabel('Data')
plt.ylabel('Cena (USD)')
plt.legend(title='Spółka')
plt.show()

# --- 3. Obliczenie i wizualizacja nr 2: Znormalizowana stopa zwrotu ---
# To jest znacznie lepszy sposób na porównanie wydajności akcji.
# Normalizujemy ceny tak, aby każda zaczynała od wartości 100.
# (cena_danego_dnia / cena_pierwszego_dnia) * 100

print("\nWizualizacja znormalizowanej stopy zwrotu (inwestycja $100)...")
normalized_returns = (data['Close'] / data['Close'].iloc[0] * 100)

normalized_returns.plot()
plt.title('Znormalizowana stopa zwrotu (początkowa inwestycja $100)')
plt.xlabel('Data')
plt.ylabel('Wartość inwestycji (USD)')
plt.legend(title='Spółka')
# Dodajemy poziomą linię na poziomie 100, aby pokazać punkt startowy
plt.axhline(y=100, color='grey', linestyle='--')
plt.show()

# --- 4. Wizualizacja nr 3: Macierz korelacji (Heatmap) ---
# Sprawdzamy, jak bardzo ceny akcji poruszają się razem.
# Wartość bliska 1 oznacza bardzo silną korelację (gdy jedna rośnie, druga też).

print("\nWizualizacja macierzy korelacji...")
# Obliczamy dzienne zmiany procentowe
daily_returns = data['Close'].pct_change()
# Obliczamy korelację
correlation_matrix = daily_returns.corr()

# Rysujemy mapę ciepła (heatmap)
sns.heatmap(correlation_matrix, 
            annot=True,        # Wyświetl wartości w komórkach
            cmap='coolwarm',   # Użyj palety niebiesko-czerwonej
            fmt=".2f",         # Formatuj liczby do 2 miejsc po przecinku
            linewidths=.5)
plt.title('Macierz korelacji dziennych stóp zwrotu')
plt.show()