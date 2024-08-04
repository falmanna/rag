from typing import Any

from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.runnables.config import RunnableConfig

from agent.nodes.base import BaseNode
from agent.state import RetrieverSubGraphState
from agent.utils.llm import get_llm
from agent.utils.misc import print_with_time


class SummarizeDocuments(BaseNode):
    @classmethod
    def get_name(cls):
        return "summarize_documents"

    @classmethod
    def get_chain(cls):
        llm = get_llm(format="")
        return LLMChainExtractor.from_llm(llm)

    @classmethod
    def invoke(
        cls, state: RetrieverSubGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        query = state.query
        documents = state.documents

        # only summarize if the query generation is enabled to save context
        if not config["configurable"].get("question_rewriter"):
            return {"documents": documents}

        print_with_time("---DOCUMENT COMPRESSOR---")
        summarized_docs = cls.get_chain().compress_documents(documents, query)

        return {"documents": summarized_docs}
