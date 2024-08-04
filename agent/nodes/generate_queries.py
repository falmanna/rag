from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.config import RunnableConfig

from agent.nodes.base import BaseNode
from agent.state import GraphState
from agent.utils.llm import get_llm
from agent.utils.misc import print_with_time


class GeneratedQueries(BaseModel):
    queries: str = Field(description="Comma separated list of generated queries")


class QueryGenerator(BaseNode):
    @classmethod
    def get_name(cls):
        return "rewriter_question"

    @classmethod
    def get_chain(cls):
        llm = get_llm()
        structured_llm_query_writer = llm.with_structured_output(GeneratedQueries)

        system = """You are an expert in rewriting questions in Arabic. \n
        Given a user's question, your task is to generate multiple queries in Arabic. \n
        These queries will be used to retrieve relevant documents from a vector database. \n
        Create distinct and clear queries for each unique concept within the user's question. \n
        Ensure each query is standalone, isolated, and does not overlap or repeat information from other queries. \n
        Avoid using vague or ambiguous references in your queries."""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Original Question: {question}"),
            ]
        )

        return prompt | structured_llm_query_writer

    @classmethod
    def invoke(cls, state: GraphState, config: RunnableConfig) -> dict[str, Any]:
        question = state.question
        if not config["configurable"].get("question_rewriter"):
            return {"queries": [question]}

        print_with_time("---GENERATE QUERIES---")
        generated_queries: GeneratedQueries = cls.get_chain().invoke(
            {"question": question}
        )

        return {"queries": generated_queries.queries.split(",")}
