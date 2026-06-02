import anthropic
from langsmith import traceable
from langsmith.wrappers import wrap_anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
model = ChatAnthropic(model="claude-haiku-4-5")
client = wrap_anthropic(anthropic.Anthropic())

@traceable(run_type="tool", name="Retrieve Context")
def my_tool(question: str) -> str:
    """
    This is a tool that should retrieve the relevant context for a given question.
    Currenly a placeholder.
    The traceable decorator passes my_tool in traceable(tool).
    traceable() logs a function's inputs and outputs as a distinct step in the overall trace.
    """
    return "During this morning's meeting, we solved all world conflict."

@traceable(name="Chat Pipeline")
def chat_pipeline(question: str):
    context = my_tool(question)
    messages = [
        SystemMessage(content="You are a helpful assistant. Please respond to the user's request only based on the given context."),
        HumanMessage(content=f"Question: {question}\nContext: {context}")
    ]
    response = model.invoke(messages)
    return response.content

chat_pipeline("Can you summarize this morning's meetings?")