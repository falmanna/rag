from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.config import RunnableConfig

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
        llm = get_llm()
        structured_llm_grader = llm.with_structured_output(HallucinationsGrade)

        system = """You are a Arabic grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary hallucination score 'true' or 'false'. \n
        'false' means that the answer is not a hallucination and is grounded in / supported by the set of facts. \n
        Explain why did you take your decision as the 'why'."""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Set of facts: \n\n {documents} \n\n LLM generation: {generation} \n\n LLM references: {references}",
                ),
            ]
        )

        return hallucination_prompt | structured_llm_grader

    @classmethod
    def invoke(cls, state: GraphState, config: RunnableConfig) -> str:
        docs = state.documents
        generation = state.generation
        references = state.references

        if not config["configurable"].get("hallucination_grader"):
            return {"hallucination_score": None}

        print_with_time("---GRADE HALLUCINATION: ANSWER vs QUESTION---")
        hallucination: HallucinationsGrade = cls.get_chain().invoke(
            {
                "documents": [doc.page_content for doc in docs],
                "generation": generation,
                "references": references,
            }
        )

        return {"hallucination_score": hallucination.binary_score}
