from typing import Any, Dict

from agent.nodes.base import BaseNode
from agent.state import GraphState
from agent.utils.misc import print_with_time
from agent.utils.retriever import get_vectorstore_retriever


class RetrieveDocs(BaseNode):
    @classmethod
    def get_name(cls) -> str:
        return "retrieve_docs"

    @classmethod
    def invoke(cls, state: GraphState) -> Dict[str, Any]:
        print_with_time("---RETRIEVE---")
        query = state.query or state.question

        documents = get_vectorstore_retriever().invoke(query)
        return {"documents": documents}
