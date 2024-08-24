import os
from typing import Any, List

from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

from agent.nodes.base import BaseNode
from agent.state import GraphState
from agent.utils.llm import get_llm
from agent.utils.misc import print_with_time


class GeneratedQueries(BaseModel):
    queries: List[str] = Field(description="list of generated queries")


class QueryGenerator(BaseNode):
    @classmethod
    def get_name(cls):
        return "rewriter_question"

    @classmethod
    def get_chain(cls):
        return get_llm()

    @classmethod
    def invoke(cls, state: GraphState, config: RunnableConfig) -> dict[str, Any]:
        question = state.question
        if not config["configurable"].get("question_rewriter"):
            return {"queries": [question]}

        print_with_time("---GENERATE QUERIES---")

        system = """You are an expert in rewriting questions in Arabic. \n
        Given a user's question, your task is to generate multiple queries in Arabic. \n
        These queries will be used to retrieve relevant documents from a vector database. \n
        Create distinct and clear queries for each unique concept within the user's question. \n
        Ensure each query is standalone, isolated, and does not overlap or repeat information from other queries. \n
        Avoid using vague or ambiguous references in your queries."""

        generated_queries: GeneratedQueries = cls.get_chain().chat.completions.create(
            model=os.environ["LLM_MODEL_NAME"],
            response_model=GeneratedQueries,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Original Question: {question}"},
            ],
        )

        return {"queries": generated_queries.queries}
