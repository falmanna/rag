from typing import Any, Dict

from langchain_core.runnables.config import RunnableConfig

from agent.nodes.base import BaseNode
from agent.state import GraphState
from agent.utils.misc import print_with_time
from agent.utils.rerank import get_reranker


class CompressDocs(BaseNode):
    @classmethod
    def get_name(cls) -> str:
        return "compress_docs"

    @classmethod
    def invoke(cls, state: GraphState, config: RunnableConfig) -> Dict[str, Any]:
        query = state.question
        docs = state.documents

        if not config["configurable"].get("question_rewriter"):
            return {"compressed_documents": docs}

        print_with_time("---COMPRESS DOCS---")
        documents = get_reranker(limit=3).compress_documents(docs, query)
        return {"compressed_documents": documents}
