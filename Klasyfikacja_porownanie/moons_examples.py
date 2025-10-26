import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report

from commons.utils import plot_decision_boundary, parse_experiment_ranges

import argparse


AVAILABLE_EXPERIMENTS = {
    1:"Regresja Logistyczna",
    2:"k-NN",
    3:"Maszyna Wektorów Nośnych (SVM) - RFB",
    
}

def main(args):
    # Parsowanie argumentów
    experiments_to_run = parse_experiment_ranges(args.experiments)
    print(f"Uruchamiam eksperymenty o numerach: {sorted(list(experiments_to_run))}")

    # 1. Generowanie i przygotowanie danych
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    print("Kształt danych wejściowych (X):", X.shape)
    print("Kształt etykiet (y):", y.shape)

    # Stworzenie DataFrame dla łatwiejszej wizualizacji
    df = pd.DataFrame(dict(x1=X[:,0], x2=X[:,1], label=y))

    # Wizualizacja zbioru danych
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='x1', y='x2', hue='label', data=df, palette='tab10')
    plt.title('Wizualizacja zbioru danych "make_moons"')
    plt.show()

    # --- Krok 2: Podział danych i skalowanie ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Skalowanie jest kluczowe dla k-NN, ale jest też dobrą praktyką dla wielu innych modeli
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.fit_transform(X)

    if 1 in experiments_to_run:
        print('------\n1. Regresja Logistyczna')
        # --- Inicjalizacja i trening modelu Regresji Logistycznej ---
        log_reg = LogisticRegression(random_state=42)
        log_reg.fit(X_train_scaled, y_train)
        # --- Predykcja i ocena ---
        y_pred_log = log_reg.predict(X_test_scaled)
        print("--- Wyniki dla Regresji Logistycznej ---")
        print(f"Dokładność (Accuracy): {accuracy_score(y_test, y_pred_log):.2f}")
        print("\nRaport klasyfikacji:")
        print(classification_report(y_test, y_pred_log))
        # --- Wizualizacja granicy decyzyjnej ---
        plot_decision_boundary(log_reg, X_train_scaled, y_train, "Granica decyzyjna - Regresja Logistyczna")

    if 2 in experiments_to_run:
        print('------\n2. k-NN')
        # --- Inicjalizacja i trening modelu k-NN ---
        # Wybierzmy k=7 jako rozsądny punkt startowy
        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(X_train_scaled, y_train)
        # --- Predykcja i ocena ---
        y_pred_knn = knn.predict(X_test_scaled)
        print("\n\n--- Wyniki dla k-Najbliższych Sąsiadów (k=7) ---")
        print(f"Dokładność (Accuracy): {accuracy_score(y_test, y_pred_knn):.2f}")
        print("\nRaport klasyfikacji:")
        print(classification_report(y_test, y_pred_knn))
        # --- Wizualizacja granicy decyzyjnej ---
        plot_decision_boundary(knn, X_train_scaled, y_train, "Granica decyzyjna - k-NN (k=7)")

    if 3 in experiments_to_run:
        print("------\n3. Maszyna Wektorów Nośnych (SVM) - RFB")
        # Używamy jądra RBF, które jest idealne do problemów nieliniowych.
        svm_clf = SVC(kernel='rbf', gamma='auto', random_state=42)
        svm_clf.fit(X_train_scaled, y_train)
        print(f"Dokładność SVM na danych testowych: {svm_clf.score(X_test_scaled, y_test):.3f}")
        y_pred_svn = svm_clf.predict(X_test_scaled)
        print(f"Dokładność (Accuracy): {accuracy_score(y_test, y_pred_svn):.2f}")
        print("\nRaport klasyfikacji:")
        print(classification_report(y_test, y_pred_svn))
        # Wizualizacja granicy decyzyjnej
        plot_decision_boundary(svm_clf, X_train_scaled, y_train, "Granica decyzyjna - SVM z jądrem RBF")



    print("\nZakończono wybrane eksperymenty.")

if __name__ == "__main__":
    experiments_help_list = "\n\nDostępne eksperymenty:\n"
    for num, desc in AVAILABLE_EXPERIMENTS.items():
        experiments_help_list += f"  {num}: {desc}\n"

    # --- Stworzenie parsera z dodatkowym tekstem pomocy (epilog) ---
    parser = argparse.ArgumentParser(
        description="Uruchamia wybrane eksperymenty klasyfikacyjne na zbiorze 'moons'.",
        # formatter_class pozwala nam używać znaków nowej linii w epilogu
        formatter_class=argparse.RawTextHelpFormatter, 
        epilog=experiments_help_list
    )
    
    # Argument --experiments pozostaje bez zmian
    parser.add_argument(
        '-e', '--experiments',
        type=str,
        required=True,
        help='Określa, które eksperymenty uruchomić. Przykłady: "2", "1,3", "1-3".'
    )
    
    # Parsowanie argumentów i wywołanie main() pozostaje bez zmian
    args = parser.parse_args()
    main(args)