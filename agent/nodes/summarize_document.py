import os
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

from agent.nodes.base import BaseNode
from agent.state import RetrieverSubGraphState
from agent.utils.llm import get_llm
from agent.utils.misc import print_with_time


class SummarizedDocument(BaseModel):
    summary: str = Field(description="Summarized content of the document")


class SummarizeDocuments(BaseNode):
    @classmethod
    def get_name(cls):
        return "summarize_documents"

    @classmethod
    def get_chain(cls):
        return get_llm()

    @classmethod
    def invoke(
        cls, state: RetrieverSubGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        query = state.query
        documents = state.documents

        if not config["configurable"].get("summarize_docs"):
            return {"documents": documents}

        print_with_time("---DOCUMENT COMPRESSOR---")
        summarized_docs = []

        system = """You are an expert in summarizing documents in Arabic. \n
        Given a document and a query, provide a concise summary of the document that is relevant to the query. \n
        Focus on the key information that answers or relates to the query."""

        for doc in documents:
            summary: SummarizedDocument = cls.get_chain().chat.completions.create(
                model=os.environ["LLM_MODEL_NAME"],
                response_model=SummarizedDocument,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": f"Document: {doc.page_content}\n\nQuery: {query}",
                    },
                ],
            )
            summarized_doc = Document(
                page_content=summary.summary, metadata=doc.metadata
            )
            summarized_docs.append(summarized_doc)

        return {"documents": summarized_docs}
