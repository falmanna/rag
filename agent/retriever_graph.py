from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from agent.nodes.grade_documents import GradeDocuments
from agent.nodes.retrieve_docs import RetrieveDocs
from agent.state import GraphConfig, RetrieverSubGraphState

load_dotenv()


def get_retriever_graph():
    graph = StateGraph(RetrieverSubGraphState, config_schema=GraphConfig)

    graph.add_node(RetrieveDocs.get_name(), RetrieveDocs.invoke)
    graph.add_node(GradeDocuments.get_name(), GradeDocuments.invoke)

    graph.set_entry_point(RetrieveDocs.get_name())
    graph.add_edge(RetrieveDocs.get_name(), GradeDocuments.get_name())
    graph.add_edge(GradeDocuments.get_name(), END)

    return graph.compile()
