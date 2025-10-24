# --- Przykład 3: Generatywna AI z plikiem .env ---
import os
from dotenv import load_dotenv # <-- 1. Importujemy funkcję

# --- Konfiguracja Kluczy API ---
# 2. Wywołujemy funkcję, która wczyta zmienne z pliku .env
# i udostępni je tak, jakby były systemowymi zmiennymi środowiskowymi.
load_dotenv() 

# Reszta kodu pozostaje DOKŁADNIE taka sama.
# Funkcja os.environ.get() teraz "widzi" zmienne załadowane z pliku .env

# --- Część A: OpenAI (model GPT) ---
try:
    from openai import OpenAI
    # Wczytanie klucza API - to działa tak jak poprzednio!
    api_key_openai = os.environ.get("OPENAI_API_KEY")
    if not api_key_openai:
        raise ValueError("Nie znaleziono klucza OPENAI_API_KEY. Upewnij się, że jest w pliku .env")

    client_openai = OpenAI(api_key=api_key_openai)

# Pobranie listy modeli
    models_list = client_openai.models.list()
    
    print("\n--- Pełna lista modeli (fragment) ---")
    # Lista może być bardzo długa, więc przejdziemy przez nią i wypiszemy najważniejsze informacje
    
    # Tworzymy słownik do grupowania modeli
    model_groups = {
        "GPT-4 (Chat)": [],
        "GPT-3.5 (Chat)": [],
        "DALL-E (Images)": [],
        "TTS (Audio)": [],
        "Whisper (Audio)": [],
        "Embeddings": [],
        "Inne": []
    }

    for model in models_list:
        model_id = model.id
        if "gpt-4" in model_id:
            model_groups["GPT-4 (Chat)"].append(model_id)
        elif "gpt-3.5" in model_id:
            model_groups["GPT-3.5 (Chat)"].append(model_id)
        elif "dall-e" in model_id:
            model_groups["DALL-E (Images)"].append(model_id)
        elif "tts" in model_id:
            model_groups["TTS (Audio)"].append(model_id)
        elif "whisper" in model_id:
            model_groups["Whisper (Audio)"].append(model_id)
        elif "embedding" in model_id:
            model_groups["Embeddings"].append(model_id)
        else:
            # Pomijamy starsze modele, żeby nie zaśmiecać listy
            if any(x in model_id for x in ['babbage', 'davinci', 'curie', 'ada']):
                continue
            model_groups["Inne"].append(model_id)

    # Wyświetlanie pogrupowanych modeli
    for group, models in model_groups.items():
        if models:
            print(f"\n--- {group} ---")
            for model_name in sorted(models):
                print(f"- {model_name}")


    prompt_uzytkownika = "Napisz haiku o programistach pythona"

    response = client_openai.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "Jesteś pomocnym asystentem, który generuje haiku na zadany temat."},
        {"role": "user", "content": prompt_uzytkownika}
      ]
    )

    print("--- Odpowiedź z OpenAI (GPT-3.5-Turbo) ---")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"Błąd OpenAI: {e}")


# --- Część B: Google (model Gemini) ---
try:
    import google.generativeai as genai
    # Wczytanie klucza API - to również działa tak jak poprzednio!
    api_key_google = os.environ.get("GOOGLE_API_KEY")
    if not api_key_google:
        raise ValueError("Nie znaleziono klucza GOOGLE_API_KEY. Upewnij się, że jest w pliku .env")
        
    genai.configure(api_key=api_key_google)
    
    for model in genai.list_models():
    # Sprawdzamy, czy dany model wspiera metodę 'generateContent'
        if 'generateContent' in model.supported_generation_methods:
            print(f"- {model.name}")

    model_gemini = genai.GenerativeModel('gemini-2.5-pro')
    
    # Ten sam prompt co dla OpenAI
    response_gemini = model_gemini.generate_content(prompt_uzytkownika)
    
    print("\n--- Odpowiedź z Google (Gemini Pro) ---")
    print(response_gemini.text)

except Exception as e:
    print(f"Błąd Gemini: {e}")