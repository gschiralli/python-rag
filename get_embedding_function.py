from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List

class CustomHuggingFaceEmbeddings:
    def __init__(self, model_name: str):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def embed_query(self, text: str):
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]):
        return self.embeddings.embed_documents(texts)

def get_embedding_function():
    # Load the pre-trained sentence-transformer model
    embeddings = CustomHuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings