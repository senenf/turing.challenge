import os

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K"))

def get_retriever(selected_docs):
    vectordb = Chroma(
        collection_name="knowledge_base",
        embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        persist_directory="db"
    )
    return vectordb.as_retriever(
        search_kwargs={"filter": {"source": {"$in": selected_docs}}, "k": RETRIEVAL_K}
    )
 