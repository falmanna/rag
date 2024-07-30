from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from agent.nodes.base import BaseNode
from agent.state import GraphState
from agent.utils.llm import get_llm
from agent.utils.misc import print_with_time
from agent.utils.parser import get_pydantic_parser


class QuestionGrade(BaseModel):
    """Route a user query to the most relevant datasource."""

    accepted: bool = Field(
        ...,
        description="Question is accepted 'true' or rejected 'false'",
    )


class GradeQuestion(BaseNode):
    @classmethod
    def get_name(cls):
        return "route_query"

    @classmethod
    def get_chain(cls):
        llm = get_llm()
        # structured_llm_router = llm.with_structured_output(QuestionGrade)
        parser = get_pydantic_parser(QuestionGrade)

        system = """You are an expert at rating users questions in Arabic. \n
        Accept questions that are BOTH in Arabic language AND can be answered by searching Arabic wikipedia. \n
        For all else, reject. \n\n
            {format_instructions}"""
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        return route_prompt | llm | parser

    @classmethod
    def invoke(cls, state: GraphState):
        print_with_time("---GRADE QUESTION---")
        question = state.question
        source: QuestionGrade = cls.get_chain().invoke({"question": question})
        return {"question_accepted": source.accepted}
