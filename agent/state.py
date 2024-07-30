from typing import Optional

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel


class Reference(BaseModel):
    reference: str
    url: Optional[str]


class GraphState(BaseModel):
    question: str
    question_accepted: Optional[bool]
    query: Optional[str]
    generation: Optional[str]
    documents: Optional[list[Document]]
    references: Optional[list[Reference]]
    hallucination_score: Optional[bool]
    usefulness_score: Optional[bool]
