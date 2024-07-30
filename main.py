from langchain.globals import set_debug, set_verbose

from agent.graph import app
from agent.utils.misc import print_with_time

set_verbose(False)
set_debug(False)


if __name__ == "__main__":
    print_with_time("Starting")
    print_with_time(app.invoke(input={"question": "متى وقعت مذبحة طبريا؟"}))

    # get_vectorstore_retriever().invoke("هل زار اينشتاين جامعة ليتكولن؟")
