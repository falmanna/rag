from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.config import RunnableConfig

from agent.nodes.base import BaseNode
from agent.state import RetrieverSubGraphState
from agent.utils.llm import get_llm
from agent.utils.misc import print_with_time
from agent.utils.parser import get_pydantic_parser


class DocumentsGrade(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    relevant: bool = Field(
        description="Documents are relevant to the query, 'true' or 'false'"
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
        # structured_llm_grader = llm.with_structured_output(DocumentsGrade)
        parser = get_pydantic_parser(DocumentsGrade)

        system = """You are an Arabic grader assessing relevance of a retrieved Arabic document to a user query. \n 
        If the document contains keyword(s) or semantic meaning close to the meaning of the query, grade it as relevant. \n
        Give a binary score 'true' or 'false' score to indicate whether the document is relevant to the query. \n
        Explain why did you take your decision as the 'why'.\n\n
        {format_instructions}"""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Retrieved documents: \n\n {documents} \n\n User query: {query}",
                ),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        return grade_prompt | llm | parser

    @classmethod
    def invoke(
        cls, state: RetrieverSubGraphState, config: RunnableConfig
    ) -> dict[str, Any]:
        query = state.query
        documents = state.documents

        if not config["configurable"].get("llm_listwise_rerank"):
            return {"documents": documents}

        print_with_time("---LISWISE RERANK: CHECK DOCUMENT RELEVANCE TO query---")
        filtered_docs = []
        for d in documents:
            score: DocumentsGrade = cls.get_chain().invoke(
                {"query": query, "documents": d.page_content}
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
