from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.config import RunnableConfig

from agent.nodes.base import BaseNode
from agent.state import GraphState
from agent.utils.llm import get_llm
from agent.utils.misc import print_with_time


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
        structured_llm_grader = llm.with_structured_output(UsefulnessGrade)

        system = """You are a Arabic grader assessing whether an answer addresses / resolves a question. \n 
        Give a binary score 'true' or 'false'. 'true' means that the answer resolves the question. \n
        Explain why did you take your decision as the 'why'."""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "User question: \n\n {question} \n\n LLM generation: {generation} \n\n LLM references: {references}",
                ),
            ]
        )

        return answer_prompt | structured_llm_grader

    @classmethod
    def invoke(cls, state: GraphState, config: RunnableConfig):
        question = state.question
        generation = state.generation
        references = state.references

        if not config["configurable"].get("usefulness_grader"):
            return {"usefulness_score": None}

        print_with_time("---GRADE USEFULNESS: ANSWER vs QUESTION---")
        grade: UsefulnessGrade = cls.get_chain().invoke(
            {"question": question, "generation": generation, "references": references}
        )
        return {"usefulness_score": grade.binary_score}
