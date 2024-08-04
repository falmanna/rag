from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.config import RunnableConfig

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
        llm = get_llm()
        structured_llm_grader = llm.with_structured_output(DocumentsGrade)

        system = """You are an Arabic grader assessing the relevance of retrieved Arabic documents to a user question. \n 
        If the document's semantic meaning closely matches the question's meaning, grade it as relevant. \n
        Provide a binary score of 'true' for relevant and 'false' for not relevant. \n
        Explain your decision as the 'why'."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Retrieved documents: \n\n {documents} \n\n User Question: {question}",
                ),
            ]
        )

        return grade_prompt | structured_llm_grader

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
        for d in documents:
            score: DocumentsGrade = cls.get_chain().invoke(
                {"question": question, "documents": d.page_content}
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
