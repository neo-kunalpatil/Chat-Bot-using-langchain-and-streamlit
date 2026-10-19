import os
from  langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm= HuggingFaceEndpoint(  repo_id="qwen2.5:1.5b")

model= ChatHuggingFace(llm = llm)
result= model.invoke("what is the capital of india")

print(result)

