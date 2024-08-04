from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from agent.nodes.base import BaseNode
from agent.state import GraphState
from agent.utils.llm import get_llm
from agent.utils.misc import print_with_time


class Generation(BaseModel):
    answer: str = Field(description="Answer to the user question")
    references: Optional[str] = Field(
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

        system = """You are an assistant specialized in answering questions in Arabic. \n
        Use the provided context to return an 'answer' the question. \n
        Keep your answer concise, using a maximum of three sentences. \n
        Cite any supporting facts as 'references'. \n
        Respond only in Arabic."""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Question: {question} \nContext: {context}"),
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
