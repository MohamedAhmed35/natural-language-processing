from pydantic import BaseModel
from typing import List
from langchain_core.documents import Document


class ChatRequest(BaseModel):
    session_id: str
    question: str

# i want to design the schema for output
class Response(BaseModel):
    status: str
    answer: str
    context: List[Document]