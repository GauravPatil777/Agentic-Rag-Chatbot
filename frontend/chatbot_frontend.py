import streamlit as st
from langchain_core.messages import BaseMessage,HumanMessage

from pathlib import Path
import sys

# Get the root project directory (Agentic_Rag_chatbot) and add it to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from backend.chatbot_backend import (
    workflow,
    generate_context,
    delete_thread,
    save_thread_name,
    retrieve_thread_name
)
import uuid
import random


st.sidebar.markdown("""
<h1 style="font-size: 44px;">🤖 AI Chatbot</h1>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<h2 style="font-size: 24px;">Upload file</h2>
""", unsafe_allow_html=True)
uploaded_file=st.sidebar.file_uploader("Choose a file",type=["pdf"])
new_chat=st.sidebar.button('New Chat')
st.sidebar.header("My coversations")

titles=["Hello what can i do for you?","Hi whats in your mind today","Hey how can assist you today?","Hey lets make a chat here","Ask whatever you want know"]

if uploaded_file:
    if (
        "uploaded_file_name" not in st.session_state
        or st.session_state.uploaded_file_name != uploaded_file.name
    ):
       with st.spinner("Processing PDF content..."):
            generate_context(uploaded_file)
            st.session_state.uploaded_file_name = uploaded_file.name
 

if "threads" not in st.session_state:
    st.session_state.threads =[]
    

if "thread_names" not in st.session_state:
    st.session_state.thread_names = retrieve_thread_name()

def generate_thread():
    return str(uuid.uuid4())

if new_chat:
    new_thread_id = str(uuid.uuid4())
    st.session_state.thread_id = new_thread_id
    st.session_state.thread_names[new_thread_id] = "New Conversation"
    save_thread_name(new_thread_id, "New Conversation")
    st.session_state.threads.append(new_thread_id)
    st.session_state.messages = []
    st.rerun()
   
if "thread_id" not in st.session_state:
    st.session_state.thread_id =str(uuid.uuid4())
    st.session_state.threads.append(st.session_state.thread_id)
    st.session_state.thread_names[st.session_state.thread_id] = "New Chat"
    save_thread_name(st.session_state.thread_id, "New Chat")

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    },
    "metadata": {
        "thread_id": st.session_state["thread_id"],
    },
     "run_name": "chatbot",
}


for thread_id in reversed(st.session_state.threads):
    
    col1, col2 = st.sidebar.columns([3, 1])

    with col1:
       
        thread_btn = st.button(st.session_state.thread_names.get(thread_id, "Unnamed Thread"), key=f"thread_{thread_id}")
        if thread_btn:
            st.session_state.thread_id = thread_id
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🗑️", key=f"delete_{thread_id}"):
            delete_thread(thread_id)
            st.session_state.threads.remove(thread_id)
            st.rerun()
state = workflow.get_state(config)

if state.values:
    st.session_state.messages=[]
    for message in state.values["messages"]:
     if message.type == "human":
        st.session_state.messages.append({
            "role": "user",
            "content": message.text
        })

     elif message.type == "ai":
        # Only display actual AI responses
        if message.content:
            st.session_state.messages.append({
                "role": "assistant",
                "content": message.text
            })
if "messages" not in st.session_state:
    st.session_state.messages = []
   
if not st.session_state.messages:
    st.title(random.choice(titles))
    
# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
st.markdown("""
<style>
[data-testid="stChatInput"] textarea {
    font-size: 20px;
    min-height: 25px;
    padding: 3px 5px;
}
</style>
""", unsafe_allow_html=True)

user_inp = st.chat_input("Type your message")

if user_inp:

    current_thread = st.session_state.thread_id

    # Give new chat a title from first message
    st.session_state.thread_names[current_thread] = user_inp[:30]
    save_thread_name(current_thread, user_inp[:30])
    st.session_state.messages.append({
        "role":"user",
        "content":user_inp
    })
    
    with st.chat_message("user"):
     with st.spinner("Thinking..."):

        st.markdown(user_inp)
        
        response=workflow.invoke(     {
                "messages": [
                    HumanMessage(content=user_inp)
                ]
            },
            config=config
        )
        
        ai_response = response['messages'][-1].text 


    # Display AI message
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response,
    })

    with st.chat_message("assistant"):
        st.markdown(ai_response)   
       