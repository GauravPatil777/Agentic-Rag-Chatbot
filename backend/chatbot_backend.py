from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import START,StateGraph
from langsmith import traceable
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.tools import tool
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
     model="gemini-3.1-flash-lite"
)

vector_store = None

def generate_context(uploaded_file):
    global vector_store
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_pdf_path = temp_file.name

    # Load PDF using the file path
    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    splitted_docs = text_splitter.split_documents(documents)

    # Generate embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # Create a vector store
    vector_store = Chroma.from_documents(splitted_docs,
                                         embedding=embeddings)
     # Delete temporary PDF
    os.remove(temp_pdf_path)
   
def retrieve_all_threads():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)
def delete_thread(thread_id):
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
search_tool=DuckDuckGoSearchRun()

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
def rag_tool(query:str)->str:
    """Search the currently uploaded PDF and answer questions using its content.
    Use this tool whenever the user asks about information contained in the uploaded PDF.
    """
    global vector_store
    if vector_store is None:
        return "No document has been uploaded."
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
        
tools=[search_tool,calculator,weather_data,rag_tool]

model_with_tools=model.bind_tools(tools)

@traceable()
def chat_node(state:chatbotState)->chatbotState:
    response=model_with_tools.invoke(state['messages'])
    return {'messages':[response]}  
    
tool_node=ToolNode(tools=tools)

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