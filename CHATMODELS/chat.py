from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()

llm= ChatGoogleGenerativeAI(model= "gemini-3.8-flash", temperature=1)

result= llm.invoke("suggest me 5 indian male names ")

print(result.content)
