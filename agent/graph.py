from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from agent.nodes.generate_answer import GenerateAnswer
from agent.nodes.grade_documents import GradeDocuments
from agent.nodes.grade_hallucination import GradeHallucinations
from agent.nodes.grade_question import GradeQuestion
from agent.nodes.grade_usefulness import GradeUsefulness
from agent.nodes.reject import Reject
from agent.nodes.retrieve_docs import RetrieveDocs
from agent.state import GraphState

load_dotenv()


def decide_to_retrieve_docs(state: GraphState):
    return "accept" if state.question_accepted else "reject"


def decide_to_generate(state: GraphState):
    return "generate" if state.documents else "reject"


workflow = StateGraph(GraphState)

workflow.add_node(GradeQuestion.get_name(), GradeQuestion.invoke)
workflow.add_node(RetrieveDocs.get_name(), RetrieveDocs.invoke)
workflow.add_node(GradeDocuments.get_name(), GradeDocuments.invoke)
workflow.add_node(GenerateAnswer.get_name(), GenerateAnswer.invoke)
workflow.add_node(GradeHallucinations.get_name(), GradeHallucinations.invoke)
workflow.add_node(GradeUsefulness.get_name(), GradeUsefulness.invoke)
workflow.add_node(Reject.get_name(), Reject.invoke)

workflow.set_entry_point(GradeQuestion.get_name())
workflow.add_conditional_edges(
    GradeQuestion.get_name(),
    decide_to_retrieve_docs,
    {
        "accept": RetrieveDocs.get_name(),
        "reject": Reject.get_name(),
    },
)

workflow.add_edge(RetrieveDocs.get_name(), GradeDocuments.get_name())
workflow.add_edge(Reject.get_name(), END)

workflow.add_conditional_edges(
    GradeDocuments.get_name(),
    decide_to_generate,
    {
        "generate": GenerateAnswer.get_name(),
        "reject": Reject.get_name(),
    },
)

workflow.add_edge(GenerateAnswer.get_name(), GradeHallucinations.get_name())
workflow.add_edge(GradeHallucinations.get_name(), GradeUsefulness.get_name())
workflow.add_edge(GradeUsefulness.get_name(), END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")
