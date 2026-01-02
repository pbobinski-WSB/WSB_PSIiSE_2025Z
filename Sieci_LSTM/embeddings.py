from dotenv import load_dotenv

load_dotenv()
import openai
import json
import numpy as np


# Funkcja do generowania osadzeń
def get_embeddings(texts, model="text-embedding-ada-002"):
    response = openai.embeddings.create(
        model=model,
        input=texts
    )
    print(response.usage.total_tokens)
    return [data.embedding for data in response.data]


def cosine_similarity_manual(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

    # Funkcja do obliczenia odległości euklidesowej
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

# Funkcja do obliczenia odległości Manhattan
def manhattan_distance(vec1, vec2):
    return np.sum(np.abs(np.array(vec1) - np.array(vec2)))

def testSimilarity(user_input = "Lubię spędzać czas na łonie natury."):

    # Lista zdań
    sentences = [
        "To jest piękny dzień.",
        "Uwielbiam spacery w parku.",
        "Czy masz ochotę na kawę?",
        "Programowanie w Pythonie jest ciekawe.",
        "Dzisiejsza pogoda jest słoneczna.",
        "Kocham książki o przygodach.",
        "Spacer w lesie to świetny relaks.",
        "Jakie plany na weekend?",
        "Film, który oglądałem wczoraj, był niesamowity.",
        "Czekam na Twój telefon."
    ]

    # Generowanie osadzeń dla listy zdań i zdania użytkownika
    all_texts = sentences + [user_input]
    embeddings = get_embeddings(all_texts)

    # Wyświetlenie wyników
    for i, embedding in enumerate(embeddings):
        print(f"Osadzenie dla tekstu {i+1}: {embedding[:5]}... (długość: {len(embedding)})")

    # Podział osadzeń na zdania z listy i osadzenie użytkownika
    sentence_embeddings = embeddings[:-1]
    user_embedding = embeddings[-1]

    print(f"Zdanie użytkownika: {user_input}")

    # Oblicz podobieństwo dla zdania użytkownika i zdań z listy
    similarities = [cosine_similarity_manual(user_embedding, sentence_embedding) for sentence_embedding in sentence_embeddings]
    print('Odległość cosinusowa')
    print(similarities)
    # Znalezienie najbardziej podobnego zdania
    most_similar_idx = np.argmax(similarities)
    most_similar_sentence = sentences[most_similar_idx]
    similarity_score = similarities[most_similar_idx]

    # Wyświetlenie wyniku
    print(f"Najbardziej podobne zdanie: {most_similar_sentence}")
    print(f"Podobieństwo: {similarity_score:.2f}")


    # Oblicz odległość euklidesową dla zdania użytkownika i zdań z listy
    distances = [euclidean_distance(user_embedding, sentence_embedding) for sentence_embedding in sentence_embeddings]
    print('Odległość euklidesowa')
    print(distances)
    # Znalezienie najbardziej podobnego zdania (UWAGA - norma odwrotna)
    most_similar_idx = np.argmin(distances)
    most_similar_sentence = sentences[most_similar_idx]
    similarity_score = distances[most_similar_idx]
    # Wyświetlenie wyniku
    print(f"Najbardziej podobne zdanie: {most_similar_sentence}")
    print(f"Podobieństwo: {similarity_score:.2f}")

    # Oblicz odległość euklidesową dla zdania użytkownika i zdań z listy
    distances = [manhattan_distance(user_embedding, sentence_embedding) for sentence_embedding in sentence_embeddings]
    print('Odległość manhatan')
    print(distances)
    # Znalezienie najbardziej podobnego zdania (UWAGA - norma odwrotna)
    most_similar_idx = np.argmin(distances)
    most_similar_sentence = sentences[most_similar_idx]
    similarity_score = distances[most_similar_idx]
    # Wyświetlenie wyniku
    print(f"Najbardziej podobne zdanie: {most_similar_sentence}")
    print(f"Podobieństwo: {similarity_score:.2f}")



if __name__ == "__main__":
    testSimilarity("Mam plan na weekend!")