import operator
from typing import Annotated, Optional

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel


class Reference(BaseModel):
    sentence: Optional[str]
    url: Optional[str]


class GraphState(BaseModel):
    question: str
    question_accepted: Optional[bool]
    queries: Optional[list[str]]
    documents: Annotated[list[Document], operator.add]
    generation: Optional[str]
    references: Optional[list[Reference]]
    hallucination_score: Optional[bool]
    usefulness_score: Optional[bool]


class GraphConfig(BaseModel):
    embedding_rerank: bool = False
    llm_listwise_rerank: bool = False
    question_rewriter: bool = False
    usefulness_grader: bool = False
    hallucination_grader: bool = False


class RetrieverSubGraphState(BaseModel):
    query: str
    documents: Optional[list[Document]]
