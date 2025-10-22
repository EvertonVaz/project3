
from pyexpat.errors import messages
from typing import List
from google import genai
from google.genai import types
from database import MessageRepository
from schemas import PromptData, RoleType

from dotenv import load_dotenv

load_dotenv()

class PersistentChatbot:

    def __init__(self):
        self.repo = MessageRepository()
        self.prompt = PromptData(
            chat_history=self.repo.get_last_n_messages(n=5),
            summary_list=self.repo.get_lasts_summary(),
            messages_count=self.repo.get_total_messages_count(),
            user_input=""
        )

    def generate_summary(self, last_n_msgs: int) -> str:
        messages = self.repo.get_last_n_messages(n=last_n_msgs)

        history_text = ''.join([f"{msg.role}: {msg.content}\n" for msg in messages])

        prompt = f"""
        <instruções>
            Resuma o seguinte histórico de conversa, seja sucinto responda em no máximo 120 palavras.
            não cite nada que não esteja no histórico e não cite os nomes dos participantes.
            deixe claro o contexto da conversa.
        </instruções>

        <historico>
            {history_text}
        </historico>
        """

        response = self.generate_llm_response(input_text=prompt, temp=1.0)

        self.repo.add_summary(content=response)
        return response


    def process_history(self) -> List[str]:
        self.prompt.chat_history = self.repo.get_last_n_messages(n=5)
        messages_count = self.repo.get_total_messages_count()

        if messages_count % 10 == 0 and messages_count > 0:
            self.summary = self.generate_summary(last_n_msgs=10)

        return self.prompt.chat_history

    def generate_chat_prompt(self, is_new_chat: bool) -> str:

        self.repo.add_message(role="user", content=self.prompt.user_input)
        self.summary_list = self.repo.get_lasts_summary()
        summarys = self.summary_list

        history_text = ''.join([f"{msg.role}: {msg.content}\n" for msg in self.prompt.chat_history])
        summary_text = ''.join([f"{summary.content}\n" for summary in summarys]) if summarys else ''

        pre_prompt = f"""
        <perfil>
        Você é um assistente conversacional inteligente, empático e prestativo.
        Características: clareza, concisão, coerência, empatia e adaptabilidade.
        </perfil>

        <instruções_principais>
        1. Responda em no máximo 25 palavras, a menos que o contexto exija mais.
        2. Mantenha tom natural, amigável e profissional
        3. Contextualize usando histórico e resumos anteriores
        4. Faça perguntas de acompanhamento quando apropriado
        5. Admita limites quando não souber algo
        {"6. NOVA CONVERSA: Ignore resumos e histórico, responda conforme contexto" if is_new_chat else "6. Aproveite o contexto anterior para continuidade temática"}
        7. Evite questionar se o usuário quer mais informações, Gostaria de saber mais? é redundante
        </instruções_principais>

        <histórico_recente>
        {history_text if history_text.strip() else "Sem histórico ainda"}
        </histórico_recente>

        <resumo_contexto>
        {summary_text if summary_text.strip() else "Sem contexto anterior"}
        </resumo_contexto>

        <entrada_usuário>
        {self.prompt.user_input}
        </entrada_usuário>

        Responda de forma natural e conversacional:
        """

        return pre_prompt


    def generate_llm_response(self, input_text: str, temp: float = 2.0) -> str:
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

    @staticmethod
    def chatbot():
        chat = PersistentChatbot()
        is_new_chat = True
        while True:
            try:
                user_input = input("Q: ")
                chat.prompt.user_input = user_input
                if user_input.lower() == 'bye':
                    break

                prompt = chat.generate_chat_prompt(is_new_chat)
                response = chat.generate_llm_response(prompt)
                print(f"A: {response}\n")

                chat.repo.add_message(role=RoleType.ASSISTANT, content=response)
                chat.process_history()
                is_new_chat = False
            except ValueError as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    PersistentChatbot.chatbot()