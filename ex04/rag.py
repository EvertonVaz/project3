import os
import pickle
from sys import argv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

load_dotenv()

FILE_CONTENT = "orbit_motordrones.txt"
EMBEDDINGS_CACHE_FILE = "embeddings_cache.pkl"
FILE_CONTENT_CACHE = "file_content_cache.pkl"

def save_embeddings_cache(embeddings, file_content):
    """Salva embeddings e conteúdo do arquivo em cache"""
    with open(EMBEDDINGS_CACHE_FILE, "wb") as f:
        pickle.dump(embeddings, f)
    with open(FILE_CONTENT_CACHE, "wb") as f:
        pickle.dump(file_content, f)
    print("✓ Cache de embeddings salvo em disco")

def load_embeddings_cache():
    """Carrega embeddings e conteúdo do cache se existirem"""
    if os.path.exists(EMBEDDINGS_CACHE_FILE) and os.path.exists(FILE_CONTENT_CACHE):
        with open(EMBEDDINGS_CACHE_FILE, "rb") as f:
            embeddings = pickle.load(f)
        with open(FILE_CONTENT_CACHE, "rb") as f:
            file_content = pickle.load(f)
        print("✓ Cache de embeddings carregado do disco")
        return embeddings, file_content
    return None, None

def get_embeddings(force_refresh: bool = False):
    """Obtém embeddings, usando cache se disponível"""
    embeddings, file_content = load_embeddings_cache()

    # Se não há cache ou força refresh, recriar embeddings
    if embeddings is None or force_refresh:
        print("Gerando embeddings...")
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        with open(FILE_CONTENT, "r", encoding="utf-8") as file:
            file_content = file.read().splitlines()

        embeddings = model.encode(file_content)
        save_embeddings_cache(embeddings, file_content)

    return embeddings, file_content

def generate_llm_response(input_text: str, temp: float = 0.4) -> str:
    if not input_text:
        raise ValueError("O prompt não pode ser vazio.")

    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=input_text,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=temp,
        )
    )

    return response.text


def similar_texts(threshold_percentage: float = 0.7) -> list[str]:
    if len(argv) < 2:
        return print("Uso: python rag.py <texto>")

    user_input = argv[1]
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    passage_embeddings, file_content = get_embeddings()

    query_embedding = model.encode(user_input)

    passage_embeddings = model.encode(file_content)

    similarity = model.similarity(query_embedding, passage_embeddings)

    max_similarity = similarity[0].max().item()

    threshold = max_similarity * threshold_percentage

    top_results = (similarity[0] >= threshold).nonzero(as_tuple=True)[0]

    similar_texts = []
    print(f"---- Linhas relevantes recuperadas: ----:")
    for idx in top_results:
        print(f"{file_content[idx]}")
        similar_texts.append(file_content[idx])
    print("---------------------------------------\n")

    return similar_texts


def main():
    similar_passages = similar_texts(0.87)
    context = "\n".join(similar_passages)

    pre_prompt = f"""
        Use o seguinte contexto para responder à pergunta do usuário de forma precisa e concisa.
        Se o contexto não for suficiente, responda que você não sabe.
        Contexto:
        {context}
        Pergunta:
        {argv[1]}
        Responda de forma natural e conversacional:
        """

    response = generate_llm_response(pre_prompt)

    return response

if __name__ == "__main__":
    print(main())