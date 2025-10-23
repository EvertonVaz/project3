from sys import argv
from sentence_transformers import SentenceTransformer

def main():
    if len(argv) < 2:
        return print("Uso: python embbedings.py <texto>")

    user_input = argv[1]
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    query_embedding = model.encode(user_input)
    frases = [
    "O cachorro correu pelo parque atrás da bola azul.",
    "Ontem a bolsa de valores fechou em queda após anúncio do governo.",
    "O vulcão entrou em erupção, iluminando o céu noturno com lava.",
    "Aprender programação em Python pode abrir muitas portas no mercado de trabalho.",
    "O café recém-moído tem um aroma que desperta memórias da infância.",
    "Cientistas descobriram uma nova espécie de peixe em águas profundas.",
    "A final da Copa foi decidida nos pênaltis, com muita emoção na torcida.",
    "O conceito de buracos negros desafia nossa compreensão do espaço-tempo.",
    "O artista usou realidade aumentada para criar uma exposição interativa.",
    "A meditação diária ajuda a reduzir o estresse e aumentar a concentração.",
    "A inteligência artificial está transformando a forma como trabalhamos e vivemos.",
    "O avanço da tecnologia tem impactado diversos setores da economia.",
    "A sustentabilidade é um tema cada vez mais relevante nas discussões globais.",
    "A educação é a chave para um futuro melhor."
    ]

    passage_embeddings = model.encode(frases)

    similarity = model.similarity(query_embedding, passage_embeddings)

    top_k = 3
    top_results = similarity[0].argsort(descending=True)[:top_k]

    print(f"\nTop {top_k} frases mais similares à query (palavra ou frase) '{user_input}':\n")
    for idx in top_results:
        print(f"{frases[idx]} (score: {similarity[0][idx]:.4f})")

    return

if __name__ == "__main__":
    main()