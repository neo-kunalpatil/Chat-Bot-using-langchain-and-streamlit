from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
import streamlit as st
model= GoogleGenerativeAI(model= "gemini-3.8-flash")
st.header("Research tool")
user_input= st.text_input("Enter your prompt ")

if st.button('Summarize'):
    result = model.invoke(user_input)
    st.write(result)

