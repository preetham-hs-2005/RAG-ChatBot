import streamlit as st
import requests

API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="Course RAG Chatbot", layout="wide")
st.title("📘 Course Notes Chatbot (Gemini RAG)")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.chat_input("Ask from your course notes...")

if user_input:
    st.session_state.chat.append(("user", user_input))

    with st.spinner("Thinking..."):
        res = requests.post(API_URL, json={"question": user_input})
        answer = res.json()["answer"]

    st.session_state.chat.append(("assistant", answer))

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg)
