from typing import Any

from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from agent.nodes.base import BaseNode
from agent.state import GraphState
from agent.utils.misc import print_with_time


class WebSearch(BaseNode):
    @classmethod
    def get_name(cls) -> str:
        return "web_search"

    @classmethod
    def invoke(cls, state: GraphState) -> dict[str, Any]:
        print_with_time("---WEB SEARCH---")
        question = state.question
        documents = state.documents

        web_search_tool = TavilySearchResults(k=3)
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]
        return {"documents": documents, "question": question}
