import uuid

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import START,StateGraph
from langsmith import traceable
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.graph.message import add_messages
import requests
import os
from dotenv import load_dotenv

load_dotenv()
model=ChatGoogleGenerativeAI(
     model="gemini-3.5-flash-lite"
)

vector_stores={}
thread_documents = {}

current_thread_id = None


def set_current_thread(thread_id):
    global current_thread_id
    current_thread_id = thread_id
def generate_context(uploaded_file,thread_id):
   
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_pdf_path = temp_file.name
    try:
        # Load PDF using the file path
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        splitted_docs = text_splitter.split_documents(documents)

        # Generate embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

        document_id = str(uuid.uuid4())
        # Create a vector store
        vector_store = Chroma.from_documents(splitted_docs,
                                             embedding=embeddings,
                                             collection_name=document_id)
        vector_stores[document_id]=vector_store
        # Map current thread to this PDF
        thread_documents[thread_id] = document_id
    finally:
         # Delete temporary PDF
            os.remove(temp_pdf_path)
   
# def retrieve_all_threads():
#     all_threads=set()
#     for checkpoint in checkpointer.list(None):
#         all_threads.add(checkpoint.config['configurable']['thread_id'])
#     return list(all_threads)
def delete_thread(thread_id):
     # Remove PDF mapping
    document_id = thread_documents.pop(thread_id, None)

    # Remove vector store
    if document_id:
        vector_stores.pop(document_id, None)
    conn = sqlite3.connect(
            "chatbot_db",
            check_same_thread=False
        )

    cursor = conn.cursor()
    # Create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS thread_names (
            thread_id TEXT PRIMARY KEY,
            thread_name TEXT
        )
    """)
    cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
     # Delete thread name
    cursor.execute(
        "DELETE FROM thread_names WHERE thread_id = ?",
        (thread_id,)
    )

    conn.commit()
    conn.close()

def save_thread_name(thread_id, thread_name):
    conn = sqlite3.connect("chatbot_db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS thread_names (
            thread_id TEXT PRIMARY KEY,
            thread_name TEXT
        )
    """)

    cursor.execute("""
        INSERT OR REPLACE INTO thread_names
        (thread_id, thread_name)
        VALUES (?, ?)
    """, (thread_id, thread_name))

    conn.commit()
    conn.close()

def retrieve_thread_name():
    conn=sqlite3.connect("chatbot_db")
    cursor=conn.cursor()
    # Create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS thread_names (
            thread_id TEXT PRIMARY KEY,
            thread_name TEXT
        )
    """)
    cursor.execute("""SELECT thread_id,thread_name from thread_names """)
    result=cursor.fetchall()
    conn.close()
    return dict(result)

class chatbotState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
    
# tools


from langchain_core.tools import tool
@tool
def web_search(query: str) -> str:
    """
    Search the web for current, recent, or time-sensitive information.
    Use this tool for news, today's information, recent events, live updates,
    current prices, current weather, and other information that may have changed.
    Do not use it for basic general-knowledge questions.
    """
    try:
        search_tool=DuckDuckGoSearchRun()
        result = search_tool.invoke(query)

        if not result:
            return "No web search results were found."

        return result

    except Exception as e:
        return (
            "WEB_SEARCH_FAILED: The web search service is currently unavailable. "
            "Do not call web_search again for this question. "
            "Answer using your existing knowledge if possible."
        )

@tool
def calculator(a: int, b: int,operation:str) -> int:
    """perform arithmetic operations on two numbers.
    supported operations: add, subtract, multiply, divide."""
    try:
        if operation == "add":
                            return a + b 
        if operation == "subtract":
                                return a - b  
        if operation == "multiply":
                                return a * b
        if operation == "divide":
                                return a / b                             
    except ZeroDivisionError:
        return "Error: Division by zero is not allowed."

@tool
def weather_data(location:str)->str:
    """Get weather data for a location."""
    url=f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={os.getenv('OPENWEATHER_API_KEY')}"
    result=requests.get(url)
    return result.json()

@tool 
def rag_tool(query: str) -> str:
    """
    Search the PDF associated with the current conversation.
    """
    global current_thread_id
    if current_thread_id is None:
        return "No conversation is currently selected."

    document_id = thread_documents.get(current_thread_id)

    if document_id is None:
        return "No PDF is associated with this conversation."

    vector_store = vector_stores.get(document_id)

    if vector_store is None:
        return "The PDF vector store could not be found."
    # perform similarity search
    similar_vectors=vector_store.similarity_search(query,3)
    context = "\n\n".join(
    doc.page_content for doc in similar_vectors
)
    prompt=PromptTemplate(
        input_variables=["context","question"],
        template="Answer the question based on the context below.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    )
    chain=prompt | model |StrOutputParser()
    result=chain.invoke({"context":context,"question":query})
    return result
        
tools=[web_search,calculator,weather_data,rag_tool]

model_with_tools=model.bind_tools(tools)
SYSTEM_PROMPT = """
You are an intelligent Agentic RAG assistant.

You have four tools:

1. rag_tool
   Use this tool whenever the user asks about the uploaded PDF,
   document, file, its contents, topics, sections, data, or information
   that may be present in the uploaded document.

2. web_search
   Use ONLY for current, recent, live, or time-sensitive information,
   such as:
   - current news
   - today's events
   - recent developments
   - current prices
   - live information
   - information that may have changed recently

   Do NOT use web_search for normal general-knowledge questions,
   basic facts, definitions, mathematics, programming concepts,
   geography, history, or other information you already know.

3. calculator
   Use for arithmetic calculations.

4. weather_data
   Use when the user asks for current weather information.

Important:
- Do not call web_search unnecessarily.
- If web_search returns WEB_SEARCH_FAILED, do not call it again.
- Instead, answer using your existing knowledge when possible.
- Give a direct and useful answer.
"""

@traceable()
def chat_node(state: chatbotState) -> chatbotState:

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"]
    ]

    response = model_with_tools.invoke(messages)

    return {"messages": [response]}
    
tool_node=ToolNode(tools=tools, handle_tool_errors=True)

# checkpointer
conn=sqlite3.connect(database='chatbot_db',check_same_thread=False)
checkpointer =SqliteSaver(conn=conn)

graph=StateGraph(chatbotState)
graph.add_node('chat_node',chat_node)
graph.add_node("tools",tool_node)
graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')


workflow=graph.compile(checkpointer=checkpointer)