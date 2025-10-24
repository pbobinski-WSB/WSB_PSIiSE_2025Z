import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Ładowanie danych (zakładamy, że plik 'diabetes.csv' jest w folderze)
# Można go pobrać stąd: https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
diabetes_df = pd.read_csv(url)

X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']

# 2. Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Skalowanie cech!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Używamy tego samego skalera!

if __name__ == "__main__":

    plt.figure(figsize=(6, 5))
    sns.countplot(x='Outcome', data=diabetes_df, palette='viridis', hue='Outcome', legend=False)
    plt.title('Rozkład klas w zbiorze danych (0 = Zdrowy, 1 = Chory na cukrzycę)')
    plt.xlabel('Wynik (Outcome)')
    plt.ylabel('Liczba pacjentów')
    plt.show()

    # Wyświetlenie dokładnych wartości
    print(diabetes_df['Outcome'].value_counts())


    # Wybierzmy kilka kluczowych cech do analizy
    features = ['Glucose', 'BMI', 'Age', 'Insulin']

    plt.figure(figsize=(15, 12))
    for i, feature in enumerate(features):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(x='Outcome', y=feature, data=diabetes_df, palette='viridis', hue='Outcome', legend=False)
        plt.title(f'Rozkład cechy "{feature}" w zależności od diagnozy')

    plt.tight_layout()
    plt.show()

    from sklearn.preprocessing import StandardScaler

    # Kopiujemy dane, żeby nie nadpisać oryginału
    X_to_scale = diabetes_df.drop('Outcome', axis=1)

    # Skalowanie
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_to_scale)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_to_scale.columns)

    # Rysowanie wykresów obok siebie
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))

    # Przed skalowaniem
    sns.violinplot(data=X_to_scale, ax=ax1, palette='viridis', orient='h')
    ax1.set_title('Dane PRZED skalowaniem (StandardScaler)')
    ax1.tick_params(axis='x', rotation=45)


    # Po skalowaniu
    sns.violinplot(data=X_scaled_df, ax=ax2, palette='viridis', orient='h')
    ax2.set_title('Dane PO skalowaniu (StandardScaler)')
    ax2.tick_params(axis='x', rotation=45)


    plt.tight_layout()
    plt.show()

    # Wykres lewy: jak różne są zakresy wartości. 
    # `Insulin` dochodzi do setek, 
    # `DiabetesPedigreeFunction` to wartości w okolicach 0-2. 
    # W algorytmie k-NN cecha `Insulin` całkowicie "zdominowałaby" obliczenia odległości.

    # Wykres prawy: Po skalowaniu wszystkie cechy mają średnią w okolicy 0 i podobną wariancję. 


