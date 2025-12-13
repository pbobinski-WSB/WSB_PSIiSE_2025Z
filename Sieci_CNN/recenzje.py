from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Model
import numpy as np
import os

# --- Przygotowanie Danych ---
max_features = 20000  # Rozmiar słownika
maxlen = 200         # Długość każdej recenzji (przycięta/dopełniona)

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# --- Definicja nazwy pliku modelu ---
model_filename = "imdb_cnn_model.keras"

# --- GŁÓWNA LOGIKA: WCZYTAJ LUB TRENUJ ---

# Sprawdzamy, czy plik z modelem już istnieje
if os.path.exists(model_filename):
    # --- ŚCIEŻKA 1: Plik istnieje - wczytujemy model ---
    print(f"--- Znaleziono istniejący model '{model_filename}'. Wczytywanie... ---")
    model_nlp = keras.models.load_model(model_filename)
    print("Model został pomyślnie wczytany.")

else:
    # --- ŚCIEŻKA 2: Plik nie istnieje - definiujemy, kompilujemy i trenujemy model ---
    print(f"--- Nie znaleziono modelu '{model_filename}'. Rozpoczynanie nowego treningu... ---")


    # --- Budowa Modelu CNN 1D ---
    model_nlp = Sequential()
    # 1. Warstwa Embedding: Zamienia numery słów na gęste wektory
    model_nlp.add(Embedding(max_features, 128, input_length=maxlen))
    # 2. Warstwa Konwolucyjna 1D: 32 filtry, każdy patrzący na "okno" 7 słów
    model_nlp.add(Conv1D(32, 7, activation='relu'))
    # 3. Warstwa Pooling: Wybiera najważniejszy sygnał z całej sekwencji
    model_nlp.add(GlobalMaxPooling1D())
    # 4. Klasyfikator
    model_nlp.add(Dense(1, activation='sigmoid'))

    

    # --- Kompilacja i Trening ---
    model_nlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_nlp.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

    print(f"\nZapisywanie nowego modelu do pliku '{model_filename}'...")
    model_nlp.save(model_filename)
    print("Model pomyślnie zapisany.")

# --- Ocena ---
model_nlp.build(input_shape=(None, maxlen))    
model_nlp.summary()
loss, accuracy = model_nlp.evaluate(x_test, y_test)
print(f"\nDokładność na danych testowych IMDB: {accuracy:.3f}")

# --- Rozwinięcie: Wizualizacja "gorących" słów w recenzji ---

# Wybierzmy jedną, przykładową recenzję do analizy (np. pierwszą ze zbioru testowego)
sample_index = 5
sample_text_encoded = x_test[sample_index]
sample_label = y_test[sample_index]

# Musimy odkodować recenzję z powrotem na słowa
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in sample_text_encoded])

print("\n--- Tworzenie modelu do wizualizacji aktywacji ---")
try:
    # Pobieramy wejście i wyjście z warstwy konwolucyjnej (layers[1]) z już wytrenowanego modelu
    activation_model = Model(inputs=model_nlp.layers[0].input, 
                         outputs=model_nlp.layers[1].output)
    print("Model aktywacji stworzony pomyślnie.")
except Exception as e:
    print(f"Wystąpił błąd przy tworzeniu modelu aktywacji: {e}")


# Pobierzmy aktywacje dla naszej recenzji
activations = activation_model.predict(np.expand_dims(sample_text_encoded, axis=0))[0]

# Uśrednijmy aktywacje po wszystkich 32 filtrach, aby uzyskać jeden "wynik ważności" dla każdego słowa
word_scores = np.mean(activations, axis=-1)

# Normalizujmy wyniki do zakresu 0-1 dla lepszej wizualizacji
word_scores = (word_scores - word_scores.min()) / (word_scores.max() - word_scores.min())

# Wyświetlenie recenzji z podświetleniem
print(f"Prawdziwy sentyment: {'Pozytywny' if sample_label == 1 else 'Negatywny'}")
print(f"Predykcja modelu: {'Pozytywny' if model_nlp.predict(np.expand_dims(sample_text_encoded, axis=0))[0][0] > 0.5 else 'Negatywny'}\n")

print("Recenzja z podświetlonymi 'gorącymi' słowami (im jaśniejszy kolor, tym ważniejsze słowo):\n")
# Do tej wizualizacji w terminalu potrzebna jest biblioteka, ale możemy zasymulować
# w prosty sposób lub przygotować HTML
# Prosta symulacja w terminalu:
words = decoded_review.split()
for i, word in enumerate(words):
    if i < len(word_scores):
        score = word_scores[i]
        if score > 0.7:
            # Użyj kodów ANSI do kolorowania w terminalu (może nie działać wszędzie)
            print(f'\x1b[48;5;220m{word}\x1b[0m', end=' ')
        elif score > 0.4:
            print(f'\x1b[48;5;226m{word}\x1b[0m', end=' ')
        else:
            print(word, end=' ')
    else:
        print(word, end=' ')



# --- Rozwinięcie Końcowe: Praktyczne Użycie Wytrenowanego Modelu ---
print('\n\nPraktyczne Użycie Wytrenowanego Modelu\n')
# Potrzebujemy słownika, aby zamienić słowa na liczby
word_index = imdb.get_word_index()

def predict_sentiment(review_text):
    """
    Funkcja przyjmuje tekst recenzji, przetwarza go i zwraca predykcję sentymentu.
    """
    print(f"Analizowana recenzja: \"{review_text}\"")
    
    # 1. Tokenizacja i konwersja na liczby całkowite
    # Zamieniamy tekst na małe litery i dzielimy na słowa
    words = review_text.lower().split()
    # Zamieniamy słowa na liczby z naszego słownika, dodając 3 (offset w słowniku IMDB)
    # Jeśli słowa nie ma w słowniku, przypisujemy 2 (oznacza "nieznane słowo")
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    
    # 2. Dopełnienie/przycięcie sekwencji do tej samej długości (maxlen)
    # Nasz model oczekuje wejścia o długości dokładnie 200
    padded_review = pad_sequences([encoded_review], maxlen=maxlen)
    
    # 3. Predykcja za pomocą naszego wytrenowanego modelu
    prediction_proba = model_nlp.predict(padded_review, verbose=0)[0][0]
    
    # 4. Interpretacja wyniku
    if prediction_proba > 0.5:
        sentiment = "Pozytywny"
    else:
        sentiment = "Negatywny"
        
    print(f"-> Wynik: Sentyment {sentiment} (Prawdopodobieństwo: {prediction_proba:.3f})\n")

# --- Przykłady użycia ---

# Przykład 1: Recenzja ewidentnie pozytywna
my_positive_review = "This was an amazing movie, one of the best I have ever seen. The acting was superb and the plot was brilliant. I highly recommend it."
predict_sentiment(my_positive_review)

# Przykład 2: Recenzja ewidentnie negatywna
my_negative_review = "I was very disappointed with this film. It was boring, the script was terrible and I almost fell asleep. A complete waste of time."
predict_sentiment(my_negative_review)

# Przykład 3: Recenzja bardziej subtelna
my_subtle_review = "The movie had some interesting ideas, but the execution was not perfect. Some scenes were good, others were just okay."
predict_sentiment(my_subtle_review)

# Przykład 4: Krótka, slangowa recenzja
my_short_review = "absolutely awful, just bad"
predict_sentiment(my_short_review)