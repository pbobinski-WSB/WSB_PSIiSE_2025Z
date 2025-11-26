import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

# Narzędzia z Keras/TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.metrics import accuracy_score, classification_report

from commons.utils import plot_decision_boundary, parse_experiment_ranges

import argparse


AVAILABLE_EXPERIMENTS = {
    1:"Regresja Logistyczna",
    2:"k-NN",
    3:"Maszyna Wektorów Nośnych (SVM) - RFB",
    4:"Przeuczone Drzewo Decyzyjne",
    5:"Przycięte Drzewo Decyzyjne",
    6:"Las Losowy",
    7:"k-Means z K=2",
    8:"Perceptron (Scikit-learn)",
    9:"MLP (Scikit-learn)",
    10:"MLP (Keras/TensorFlow)"
    
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

    if 4 in experiments_to_run:
        print("------\n4. Przeuczone Drzewo Decyzyjne")
        tree_overfit = DecisionTreeClassifier(random_state=42)
        tree_overfit.fit(X_train_scaled, y_train)
        # Porównujemy wyniki na zbiorze treningowym i testowym
        print(f"Dokładność na zbiorze TRENINGOWYM: {tree_overfit.score(X_train_scaled, y_train):.3f}")
        print(f"Dokładność na zbiorze TESTOWYM:    {tree_overfit.score(X_test_scaled, y_test):.3f}")
        y_pred_tree = tree_overfit.predict(X_test_scaled)
        print(f"Dokładność (Accuracy): {accuracy_score(y_test, y_pred_tree):.2f}")
        print("\nRaport klasyfikacji:")
        print(classification_report(y_test, y_pred_tree))
        # Wizualizacja - zobaczymy "poszarpaną" granicę
        plot_decision_boundary(tree_overfit, X_train_scaled, y_train, "Granica decyzyjna - Przeuczone Drzewo Decyzyjne")

    if 5 in experiments_to_run:
        print("------\n5. Przycięte Drzewo Decyzyjne")
        # Ograniczamy głębokość drzewa, aby zapobiec przeuczeniu i poprawić generalizację.
        tree_pruned = DecisionTreeClassifier(max_depth=5, random_state=42)
        tree_pruned.fit(X_train_scaled, y_train)
        print(f"Dokładność na zbiorze TRENINGOWYM: {tree_pruned.score(X_train_scaled, y_train):.3f}")
        print(f"Dokładność na zbiorze TESTOWYM:    {tree_pruned.score(X_test_scaled, y_test):.3f}")
        y_pred_treep = tree_pruned.predict(X_test_scaled)
        print(f"Dokładność (Accuracy): {accuracy_score(y_test, y_pred_treep):.2f}")
        print("\nRaport klasyfikacji:")
        print(classification_report(y_test, y_pred_treep))
        # Wizualizacja - granica będzie "schodkowa", ale bardziej regularna
        plot_decision_boundary(tree_pruned, X_train_scaled, y_train, "Granica decyzyjna - Drzewo (max_depth=5)")

    if 6 in experiments_to_run:
        print("------\n6. Las Losowy")
        # Używamy wielu drzew, aby uzyskać jeszcze lepszy i gładszy model.
        # n_estimators=100 oznacza, że nasz las będzie się składał ze 100 drzew
        forest = RandomForestClassifier(n_estimators=100, random_state=42)
        forest.fit(X_train_scaled, y_train)
        print(f"Dokładność na zbiorze TRENINGOWYM: {forest.score(X_train_scaled, y_train):.3f}")
        print(f"Dokładność na zbiorze TESTOWYM:    {forest.score(X_test_scaled, y_test):.3f}")
        y_pred_forest = forest.predict(X_test_scaled)
        print(f"Dokładność (Accuracy): {accuracy_score(y_test, y_pred_forest):.2f}")
        print("\nRaport klasyfikacji:")
        print(classification_report(y_test, y_pred_forest))
        # Wizualizacja - granica będzie znacznie gładsza i bardziej stabilna
        plot_decision_boundary(forest, X_train_scaled, y_train, "Granica decyzyjna - Las Losowy")

    if 7 in experiments_to_run:
        print("------\n7. k-Means z K=2")
        # Budujemy model k-Means z K=2
        kmeans_moons = KMeans(n_clusters=2, random_state=42, n_init=10)
        predicted_clusters = kmeans_moons.fit_predict(X_scaled)
        # Pobranie współrzędnych centroid z wytrenowanego modelu
        centroids_moons = kmeans_moons.cluster_centers_
        # Wyświetlenie współrzędnych centroid (opcjonalnie, dla ciekawości)
        print("Współrzędne znalezionych centroid:")
        print(centroids_moons)
        # Wizualizacja wyników
        plt.figure(figsize=(10, 7))
        # Rysowanie punktów danych pokolorowanych według znalezionego klastra
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=predicted_clusters, 
                    cmap='viridis', s=50, label='Dane')
        plt.scatter(centroids_moons[:, 0], centroids_moons[:, 1], s=300, c='red', 
                    marker='X', label='Centroidy')
        plt.title('Wynik działania k-Means na zbiorze "moons" (z widocznymi centroidami)')
        plt.xlabel("Cecha 1 (skalowana)")
        plt.ylabel("Cecha 2 (skalowana)")
        plt.legend()
        plt.grid(True)
        plt.show()


    if 8 in experiments_to_run:
        print("------\n8. Perceptron (Scikit-learn)")
        # --- Perceptron (model liniowy) ---
        perceptron = Perceptron(random_state=42)
        perceptron.fit(X_train_scaled, y_train)
        print(f"Dokładność Perceptronu na danych testowych: {perceptron.score(X_test_scaled, y_test):.3f}")
        y_pred_perc = perceptron.predict(X_test_scaled)
        print(f"Dokładność (Accuracy): {accuracy_score(y_test, y_pred_perc):.2f}")
        print("\nRaport klasyfikacji:")
        print(classification_report(y_test, y_pred_perc))
        plot_decision_boundary(perceptron, X_train_scaled, y_train, "Granica decyzyjna - Perceptron")

    if 9 in experiments_to_run:
        print("------\n9. MLP (Scikit-learn)")
        # --- Perceptron Wielowarstwowy (MLP w Scikit-learn) ---
        mlp_sklearn = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
        mlp_sklearn.fit(X_train_scaled, y_train)
        print(f"Dokładność MLP (Scikit-learn) na danych testowych: {mlp_sklearn.score(X_test_scaled, y_test):.3f}")
        y_pred_mlp = mlp_sklearn.predict(X_test_scaled)
        print(f"Dokładność (Accuracy): {accuracy_score(y_test, y_pred_mlp):.2f}")
        print("\nRaport klasyfikacji:")
        print(classification_report(y_test, y_pred_mlp))
        plot_decision_boundary(mlp_sklearn, X_train_scaled, y_train, "Granica decyzyjna - MLP (Scikit-learn)")

    if 10 in experiments_to_run:
        print("------\n10. MLP (Keras/TensorFlow)")
        # --- Perceptron Wielowarstwowy (MLP w Keras/TensorFlow) ---

        import tensorflow as tf
        tf.random.set_seed(42)

        # 1. Definicja modelu
        model_keras = Sequential([
            Dense(16, activation='relu', input_shape=(2,)),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
            # Dense(10, activation='relu', input_shape=(2,)),
            # Dense(5, activation='relu'),
            # Dense(1, activation='sigmoid')
        ])

        # 2. Kompilacja modelu
        model_keras.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # 3. Trening modelu
        model_keras.fit(X_train_scaled, y_train, epochs=100, verbose=0)
        # print("Rozpoczynanie dłuższego treningu...")
        # model_keras.fit(X_train_scaled, y_train, epochs=500, verbose=0, batch_size=32)
        
        # 4. Ocena modelu
        loss, accuracy = model_keras.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"Dokładność MLP (Keras) na danych testowych: {accuracy:.3f}")
        y_pred_keras = (model_keras.predict(X_test_scaled, verbose=0) > 0.5).astype(int)
        print(f"Dokładność (Accuracy): {accuracy_score(y_test, y_pred_keras):.2f}")        
        print("\nRaport klasyfikacji - MLP (Keras):")
        print(classification_report(y_test, y_pred_keras))

        # 5. Wizualizacja - potrzebujemy małej klasy "opakowującej" model Keras
        class KerasModelWrapper:
            def __init__(self, model):
                self.model = model
            def predict(self, X):
                return (self.model.predict(X, verbose=0) > 0.5).astype(int).flatten()

        plot_decision_boundary(KerasModelWrapper(model_keras), X_train_scaled, y_train, "Granica decyzyjna - MLP (Keras)")


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