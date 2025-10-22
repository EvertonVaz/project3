from enum import Enum
from pydantic import BaseModel
from typing import Optional

class RoleType(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class MessageData(BaseModel):
    role: RoleType
    content: str

class SummaryData(BaseModel):
    content: str

class PromptData(BaseModel):
    chat_history: list[MessageData]
    summary_list: list[SummaryData]
    messages_count: int
    user_input: str