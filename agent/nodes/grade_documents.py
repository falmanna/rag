import os
from typing import Any

from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field

from agent.nodes.base import BaseNode
from agent.state import RetrieverSubGraphState
from agent.utils.llm import get_llm
from agent.utils.misc import print_with_time


class DocumentsGrade(BaseModel):
    relevant: bool = Field(
        description="Documents are relevant to the question, 'true' or 'false'"
    )
    why: str = Field(
        description="Reasoning for the relevance score",
    )


class GradeDocuments(BaseNode):
    @classmethod
    def get_name(cls):
        return "grade_documents"

    @classmethod
    def get_chain(cls):
        return get_llm()

    @classmethod
    def invoke(
        cls, state: RetrieverSubGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        question = state.original_question
        documents = state.documents

        if not config["configurable"].get("llm_rerank"):
            return {"documents": documents}

        print_with_time("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        filtered_docs = []

        system = """You are an Arabic grader assessing the relevance of retrieved Arabic documents to a user question. \n 
        If the document's semantic meaning closely matches the question's meaning, grade it as relevant. \n
        Provide a binary score of 'true' for relevant and 'false' for not relevant. \n
        Explain your decision as the 'why'."""

        for d in documents:
            score: DocumentsGrade = cls.get_chain().chat.completions.create(
                model=os.environ["LLM_MODEL_NAME"],
                response_model=DocumentsGrade,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": f"Retrieved documents: \n\n {d.page_content} \n\n User Question: {question}",
                    },
                ],
            )
            if score.relevant:
                print_with_time(
                    f"---GRADE: DOCUMENT {d.metadata.get('title')} RELEVANT---"
                )
                filtered_docs.append(d)
            else:
                print_with_time(
                    f"---GRADE: DOCUMENT {d.metadata.get('title')} IRRELEVANT---"
                )

        return {"documents": filtered_docs}
