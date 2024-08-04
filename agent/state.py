from typing import Annotated, Optional

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel


def merge_and_deduplicate_lists(x: list[Document], y: list[Document]):
    # Merge the lists and remove duplicates
    combined = x + y
    seen = set()
    deduplicated = []
    for item in combined:
        if (item.metadata["id"], item.metadata["start_index"]) not in seen:
            deduplicated.append(item)
            seen.add((item.metadata["id"], item.metadata["start_index"]))
    return deduplicated


class GraphState(BaseModel):
    question: str
    question_accepted: Optional[bool]
    queries: Optional[list[str]]
    documents: Annotated[list[Document], merge_and_deduplicate_lists]
    generation: Optional[str]
    references: Optional[list[str]]
    hallucination_score: Optional[bool]
    usefulness_score: Optional[bool]


class RetrieverSubGraphState(BaseModel):
    original_question: str
    query: str
    documents: Optional[list[Document]]


class GraphConfig(BaseModel):
    embedding_rerank: bool = False
    llm_rerank: bool = False
    question_rewriter: bool = False
    usefulness_grader: bool = False
    hallucination_grader: bool = False
