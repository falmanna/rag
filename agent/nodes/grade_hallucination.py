import os

from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

from agent.nodes.base import BaseNode
from agent.state import GraphState
from agent.utils.llm import get_llm
from agent.utils.misc import print_with_time


class HallucinationsGrade(BaseModel):
    binary_score: bool = Field(
        description="Answer is a hallucination 'true' or grounded in the facts 'false'"
    )
    why: str = Field(description="Reasoning for the score")


class GradeHallucinations(BaseNode):
    @classmethod
    def get_name(cls):
        return "hallucination_grade"

    @classmethod
    def get_chain(cls):
        return get_llm()

    @classmethod
    def invoke(cls, state: GraphState, config: RunnableConfig) -> str:
        docs = state.documents
        generation = state.generation
        references = state.references

        if not config["configurable"].get("hallucination_grader"):
            return {"hallucination_score": None}

        print_with_time("---GRADE HALLUCINATION: ANSWER vs QUESTION---")

        system = """You are an Arabic grader assessing whether an LLM generation is supported by a set of retrieved facts. \n 
        Give a binary score: 'true' or 'false'. \n
        'false' means the answer is not a hallucination and is supported by the facts. \n
        Explain your decision as the 'why'."""

        hallucination: HallucinationsGrade = cls.get_chain().chat.completions.create(
            model=os.environ["LLM_MODEL_NAME"],
            response_model=HallucinationsGrade,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Set of facts: \n\n {[doc.page_content for doc in docs]} \n\n LLM generation: {generation} \n\n LLM references: {references}",
                },
            ],
        )

        return {"hallucination_score": hallucination.binary_score}
