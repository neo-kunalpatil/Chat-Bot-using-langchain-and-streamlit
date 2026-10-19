from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()
import streamlit as st
model= GoogleGenerativeAI(model= "gemini-3.8-flash")
st.header("Research tool")
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )
technique_input=  st.selectbox("Select example form", ["One-shot", "Few-Shot", "Zero-Shot"])




from pathlib import Path

template_path = Path(__file__).resolve().parent / "template.json"
template = load_prompt(str(template_path))





if st.button("Summarize"):
    examples = ""

    if technique_input == "One-shot":
        examples = """
Example:

Paper: Attention Is All You Need
Style: Beginner-Friendly
Example explanation:
The Transformer is a neural network architecture
that uses attention to understand relationships
between words. It processes relationships between
tokens without relying on recurrent processing.

Use a similar clarity and structure for the selected paper.
"""

    elif technique_input == "Few-shot":
        examples = """
Example 1:
Paper: Attention Is All You Need
Style: Beginner-Friendly
Output: Explain attention and the Transformer using
simple language and a real-world analogy.

Example 2:
Paper: BERT
Style: Technical
Output: Explain bidirectional context, pretraining,
and the model's encoder architecture.

Example 3:
Paper: GPT-3
Style: Code-Oriented
Output: Explain few-shot learning and illustrate
how examples can guide a model's predictions.

Use these examples to guide the response style.
Do not copy their paper-specific details into another paper.
"""

    chain = template | model

    result = chain.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input,
        "technique_input": technique_input,
        "examples": examples
    })

    st.write(result)
