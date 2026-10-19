from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbedding
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
import streamlit as st



load_dotenv()
model = GoogleGenerativeAI(model= "gemini-3.8-flash")

st.header("welcome")
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )


temp = PromptTemplate(
    template="""
You are a research paper assistant.

Explain the following research paper:
Paper Name: {paper_input}

Explanation Style: {style_input}

Explanation Length: {length_input}

Instructions:
1. Explain the main idea of the paper.
2. Describe the problem the paper tries to solve.
3. Explain its key methodology and architecture.
4. Mention important results or contributions.
5. Adapt the explanation to the selected style and length.
6. Use simple language when the Beginner-Friendly style is selected.

Generate the explanation now.
""",
    input_variables=["paper_input", "style_input", "length_input"]
)

prompt = temp.invoke( {"paper_input" :paper_input,
                "style_input":style_input,
               "length_input":length_input})  

if st.button("summarize"):
    result= model.invoke(prompt) 
    st.write(result) 




