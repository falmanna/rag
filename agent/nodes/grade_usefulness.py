from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from agent.nodes.base import BaseNode
from agent.state import GraphState
from agent.utils.llm import get_llm
from agent.utils.misc import print_with_time
from agent.utils.parser import get_pydantic_parser

NAME = "answer_grade"


class UsefulnessGrade(BaseModel):
    binary_score: bool = Field(
        description="Answer addresses the question, 'true' or 'false'"
    )
    why: str = Field(description="Reasoning for the score")


class GradeUsefulness(BaseNode):
    @classmethod
    def get_name(cls):
        return "usefulness_grade"

    @classmethod
    def get_chain(cls):
        llm = get_llm()
        # structured_llm_grader = llm.with_structured_output(UsefulnessGrade)
        parser = get_pydantic_parser(UsefulnessGrade)

        system = """You are a Arabic grader assessing whether an answer addresses / resolves a question. \n 
        Give a binary score 'true' or 'false'. 'true' means that the answer resolves the question. \n
        Explain why did you take your decision as the 'why'.\n\n
        {format_instructions}"""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "User question: \n\n {question} \n\n LLM generation: {generation} \n\n LLM references: {references}",
                ),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        return answer_prompt | llm | parser

    @classmethod
    def invoke(cls, state: GraphState):
        print_with_time("---GRADE USEFULNESS: ANSWER vs QUESTION---")
        question = state.question
        generation = state.generation
        references = state.references

        grade: UsefulnessGrade = cls.get_chain().invoke(
            {"question": question, "generation": generation, "references": references}
        )
        return {"usefulness_score": grade.binary_score}
