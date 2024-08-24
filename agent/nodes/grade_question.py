import os

from pydantic import BaseModel, Field

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
        return get_llm()

    @classmethod
    def invoke(cls, state: GraphState):
        print_with_time("---GRADE QUESTION---")
        question = state.question

        system = """You are an expert at rating users questions in Arabic. \n
        Give a binary score 'accepted' as true or false. \n
        true means the question is in BOTH in Arabic language AND can be answered by searching Arabic wikipedia. \n
        For all else, false."""

        source: QuestionGrade = cls.get_chain().chat.completions.create(
            model=os.environ["LLM_MODEL_NAME"],
            response_model=QuestionGrade,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
        )
        return {"question_accepted": source.accepted}
