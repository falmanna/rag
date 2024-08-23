from typing import Any, Dict

from langchain_core.runnables.config import RunnableConfig

from agent.nodes.base import BaseNode
from agent.state import RetrieverSubGraphState
from agent.utils.misc import print_with_time
from agent.utils.retriever import get_retriever


class RetrieveDocs(BaseNode):
    @classmethod
    def get_name(cls) -> str:
        return "retrieve_docs"

    @classmethod
    def invoke(
        cls, state: RetrieverSubGraphState, config: RunnableConfig
    ) -> Dict[str, Any]:
        print_with_time("---RETRIEVE---")
        query = state.query

        rerank: bool = config["configurable"].get("embedding_rerank")
        documents = get_retriever(rerank=rerank, limit=3).invoke(query)
        return {"documents": documents}
