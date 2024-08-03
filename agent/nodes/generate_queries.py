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

        system = """You are an Arabic question rewriting expert. \n
        Based on the user question, your task is to generate a single or multiple queries in Arabic \n
        The queries will be used to retrieve relevant documents from a vector database. \n
        You should generate different queries for different concepts in the user's question."""

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
