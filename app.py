import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

from langgraph.prebuilt import tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = openai_api_key

from Modules.retriever import textbook_retriever_tool, summary_retriever_tool

tools = [summary_retriever_tool, textbook_retriever_tool]  # Add the new tool to the tools list

# Define the agent state
from typing import Annotated, Literal, Sequence, TypedDict
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# Want to look into expanding the agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Define agent functions
def agent(state):
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    return {"messages": [response]}

def rewrite(state):
    messages = state["messages"]
    question = messages[0].content
    msg = [HumanMessage(content=f"Reformulate the question: {question}")]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
    response = model.invoke(msg)
    return {"messages": [response]}

def generate(state):
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

def grade_documents(state) -> Literal["generate", "rewrite"]:
    class grade(BaseModel):
        binary_score: str = Field(description="Relevance score: 'yes' or 'no'")
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    llm_with_tool = model.with_structured_output(grade)
    prompt = PromptTemplate(
        template="""Assess if the document is relevant to the user's question:
                    Document: {context}
                    Question: {question}
                    Answer 'yes' or 'no'.""",
        input_variables=["context", "question"]
    )
    chain = prompt | llm_with_tool
    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content
    result = chain.invoke({"question": question, "context": docs})
    return "generate" if result.binary_score == "yes" else "rewrite"

# Need to define a node that grades a student response, that adheres to a specific format
# Maybe define two different ones for MCQ and SAQ.

# Need to define a node that evaluates whether a student is ready to advance

# Build the workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
retrieve = ToolNode(tools)
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile the workflow
graph = workflow.compile()

# Example usage
import pprint

inputs = {
    "messages": [
        ("user", "Summarize chapter 1"),
    ]
}
for output in graph.stream(inputs):
    for key, value in output.items():
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint(value)
