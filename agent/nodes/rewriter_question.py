from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from agent.nodes.base import BaseNode
from agent.state import GraphState
from agent.utils.llm import get_llm
from agent.utils.misc import print_with_time
from agent.utils.parser import get_pydantic_parser


class QuestionQuery(BaseModel):
    value: str = Field(
        description="Generated query value to improve retrieval from a vector database"
    )


class QuestionRewriter(BaseNode):
    @classmethod
    def get_name(cls):
        return "rewriter_question"

    @classmethod
    def get_chain(cls):
        llm = get_llm(temperature=0.1)
        # structured_llm_query_writer = llm.with_structured_output(QuestionQuery)
        parser = get_pydantic_parser(QuestionQuery)

        system = """You are an Arabic question rewriting expert. \n
            Based on the user question, your task is to generate a query in Arabic to retrieve relevant documents from a vector database. \n
            By generating a better query, your goal is to help the user overcome some of the limitations of the distance-based similarity search. \n
            Provide the new query as 'value'.\n\n
            {format_instructions}"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Original Question: {question}"),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        return prompt | llm | parser

    @classmethod
    def generate_query(cls, state: GraphState) -> dict[str, Any]:
        print_with_time("---GENERATE IMPROVED QUERY---")
        question = state.question

        query: QuestionQuery = cls.get_chain().invoke({"question": question})

        return {"query": query.value}
