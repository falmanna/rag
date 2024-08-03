from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from agent.nodes.base import BaseNode
from agent.state import GraphState
from agent.utils.llm import get_llm
from agent.utils.misc import print_with_time


class QuestionGrade(BaseModel):
    accepted: bool = Field(
        description="Question is accepted 'true' or rejected 'false'",
    )


class GradeQuestion(BaseNode):
    @classmethod
    def get_name(cls):
        return "grade_question"

    @classmethod
    def get_chain(cls):
        llm = get_llm()
        structured_llm_router = llm.with_structured_output(QuestionGrade)

        system = """You are an expert at rating users questions in Arabic. \n
        Give a binary score 'accepted' as true or false. \n
        true means the question is in BOTH in Arabic language AND can be answered by searching Arabic wikipedia. \n
        For all else, false."""
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )

        return route_prompt | structured_llm_router

    @classmethod
    def invoke(cls, state: GraphState):
        print_with_time("---GRADE QUESTION---")
        question = state.question
        source: QuestionGrade = cls.get_chain().invoke({"question": question})
        return {"question_accepted": source.accepted}
