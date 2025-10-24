from typing import List
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def generate_llm_response(input_text: str, temp: float = 2.0) -> str:
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

def process_history(input_text: str, history: List[str] = []) -> List[str]:
    history.append(input_text+"\n\n")

    if len(history) > 5:
        history = history[-5:]

    return history

def generate_chat_prompt(user_input: str, history: List[str]) -> str:
    pre_prompt = f"Responda de forma concisa e clara. em no maximo 10 palavras.\n\n"
    pre_prompt += "Aqui está o histórico da conversa até agora:\n"
    pre_prompt += "".join(history)
    pre_prompt += f"Q: {user_input}\nA:"
    return pre_prompt

def chatbot():
    history = []
    while True:
        try:
            user_input = input("Q: ")
            if user_input.lower() == 'bye':
                break
            prompt = generate_chat_prompt(user_input, history)
            response = generate_llm_response(prompt)
            print(f"A: {response}\n")
            history = process_history(f"Q: {user_input}\nA: {response}", history)
        except ValueError as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    chatbot()