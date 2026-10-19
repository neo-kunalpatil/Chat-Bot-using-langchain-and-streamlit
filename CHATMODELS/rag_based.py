
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=300
)

document = [
    "Machine learning enables computers to learn from data.",
    "Deep learning uses neural networks with multiple layers.",
    "LangChain helps developers build applications using large language models.",
    "Retrieval-Augmented Generation combines retrieved documents with AI-generated answers.",
    "Embeddings convert text into numerical vectors for semantic search."
]

query = "How does Retrieval-Augmented Generation use embeddings and retrieved documents to generate accurate answers?"

doc_embedding = embedding.embed_documents(document)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)[0]

# Show similarity scores, highest first
ranked_results = sorted(
    enumerate(scores),
    key=lambda item: item[1],
    reverse=True
)

for index, score in ranked_results:
    print(f"Score: {score:.4f} | Document: {document[index]}")
