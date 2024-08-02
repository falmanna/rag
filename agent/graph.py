from dotenv import load_dotenv
from langgraph.constants import Send
from langgraph.graph import END, StateGraph

from agent.nodes.generate_answer import GenerateAnswer
from agent.nodes.generate_queries import QueryGenerator
from agent.nodes.grade_hallucination import GradeHallucinations
from agent.nodes.grade_question import GradeQuestion
from agent.nodes.grade_usefulness import GradeUsefulness
from agent.nodes.reject import Reject
from agent.retriever_graph import get_retriever_graph
from agent.state import GraphConfig, GraphState, RetrieverSubGraphState

load_dotenv()


def decide_to_accept_question(state: GraphState):
    return "accept" if state.question_accepted else "reject"


def decide_to_generate(state: GraphState):
    return "generate" if state.documents else "reject"


def send_queries(state: GraphState):
    return [
        Send("retriever_subgraph", RetrieverSubGraphState(query=query))
        for query in state.queries
    ]


def get_main_graph():
    graph = StateGraph(GraphState, config_schema=GraphConfig)

    graph.add_node(GradeQuestion.get_name(), GradeQuestion.invoke)
    graph.add_node(QueryGenerator.get_name(), QueryGenerator.invoke)
    graph.add_node(GenerateAnswer.get_name(), GenerateAnswer.invoke)
    graph.add_node(GradeHallucinations.get_name(), GradeHallucinations.invoke)
    graph.add_node(GradeUsefulness.get_name(), GradeUsefulness.invoke)
    graph.add_node(Reject.get_name(), Reject.invoke)
    graph.add_node("retriever_subgraph", get_retriever_graph())

    graph.set_entry_point(GradeQuestion.get_name())
    graph.add_edge(Reject.get_name(), END)

    # Grade question
    graph.add_conditional_edges(
        GradeQuestion.get_name(),
        decide_to_accept_question,
        {
            "accept": QueryGenerator.get_name(),
            "reject": Reject.get_name(),
        },
    )

    # Map-reduce retriever subgraph
    graph.add_conditional_edges(
        QueryGenerator.get_name(), send_queries, ["retriever_subgraph"]
    )

    graph.add_conditional_edges(
        "retriever_subgraph",
        decide_to_generate,
        {
            "generate": GenerateAnswer.get_name(),
            "reject": Reject.get_name(),
        },
    )

    graph.add_edge(GenerateAnswer.get_name(), GradeHallucinations.get_name())
    graph.add_edge(GenerateAnswer.get_name(), GradeUsefulness.get_name())
    graph.add_edge(GradeUsefulness.get_name(), END)
    graph.add_edge(GradeHallucinations.get_name(), END)

    return graph.compile()
