
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.title("Kunal ChatBot")

model = GoogleGenerativeAI(model="gemini-3.8-flash")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display complete chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Message Kunal ChatBot..."):

    # Save and display user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = model.invoke(prompt)
                st.markdown(response)

                # Save assistant response
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                st.error(f"Error: {e}")
