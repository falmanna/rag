import os
from typing import Optional

from pydantic import BaseModel, Field

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
        return get_llm()

    @classmethod
    def invoke(cls, state: GraphState):
        print_with_time("---GENERATE---")
        question = state.question
        docs = state.documents
        context = "\n\n".join([doc.page_content for doc in docs])

        system = """You are an assistant specialized in answering questions in Arabic. \n
        Use the provided context to return an 'answer' the question. \n
        Keep your answer concise, using a maximum of three sentences. \n
        Cite any supporting facts as 'references'. \n
        Respond only in Arabic."""

        generation: Generation = cls.get_chain().chat.completions.create(
            model=os.environ["LLM_MODEL_NAME"],
            response_model=Generation,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"Question: {question} \nContext: {context}",
                },
            ],
        )

        return {
            "generation": generation.answer,
            "references": generation.references,
        }
