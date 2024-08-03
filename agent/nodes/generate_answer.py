from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from agent.nodes.base import BaseNode
from agent.state import GraphState
from agent.utils.llm import get_llm
from agent.utils.misc import print_with_time


class Reference(BaseModel):
    sentence: Optional[str] = Field(
        description="Cite sentence that support your decision from the context (if any)"
    )
    url: Optional[str] = Field(description="URL of the reference (if any)")


class Generation(BaseModel):
    answer: str = Field(description="Answer to the user question")
    references: Optional[list[Reference]] = Field(
        description="Cite the fact that support your decision from the context (if any)"
    )


class GenerateAnswer(BaseNode):
    @classmethod
    def get_name(cls):
        return "generate_answer"

    @classmethod
    def get_chain(cls):
        llm = get_llm()
        structured_llm_query_writer = llm.with_structured_output(Generation)

        system = """You are an assistant for question-answering tasks in Arabic. \n
        Use the following pieces of retrieved context to answer the question. \n
        If you don't know the answer, just say that you don't know. \n
        Use three sentences maximum and keep the answer concise. \n
        Cite the facts that support your decision (if any) as the 'references'. \n
        Answer in Arabic language only."""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Question: {question} \n\n  Context: {context}"),
            ]
        )

        return prompt | structured_llm_query_writer

    @classmethod
    def invoke(cls, state: GraphState):
        print_with_time("---GENERATE---")
        question = state.question
        docs = state.documents

        generation: Generation = cls.get_chain().invoke(
            {
                "context": "\n\n".join([doc.page_content for doc in docs]),
                "question": question,
            }
        )

        return {
            "generation": generation.answer,
            "references": generation.references,
        }
