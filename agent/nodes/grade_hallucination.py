from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from agent.nodes.base import BaseNode
from agent.state import GraphState
from agent.utils.llm import get_llm
from agent.utils.misc import print_with_time
from agent.utils.parser import get_pydantic_parser


class HallucinationsGrade(BaseModel):
    """Binary score for hallucination present in generation answer."""

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
        # structured_llm_grader = llm.with_structured_output(HallucinationsGrade)
        parser = get_pydantic_parser(HallucinationsGrade)

        system = """You are a Arabic grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
            Give a binary hallucination score 'true' or 'false'. \n
            'false' means that the answer is not a hallucination and is grounded in / supported by the set of facts. \n
            Explain why did you take your decision as the 'why'. \n\n
            {format_instructions}"""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Set of facts: \n\n {documents} \n\n LLM generation: {generation} \n\n LLM references: {references}",
                ),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        return hallucination_prompt | llm | parser

    @classmethod
    def invoke(cls, state: GraphState) -> str:
        print_with_time("---GRADE HALLUCINATION: ANSWER vs QUESTION---")
        documents = state.documents
        generation = state.generation
        references = state.references

        hallucination: HallucinationsGrade = cls.get_chain().invoke(
            {"documents": documents, "generation": generation, "references": references}
        )

        return {"hallucination_score": hallucination.binary_score}
