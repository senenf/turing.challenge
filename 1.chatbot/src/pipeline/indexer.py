import os 
import json

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv("utils/.env")

# Load API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY in environment variables")

# Load and parse embedding model config
embedding_json = os.getenv("EMBEDDING_MODELS_JSON")
if not embedding_json:
    raise EnvironmentError("Missing EMBEDDING_MODELS_JSON in environment variables")

embedding_models = json.loads(embedding_json)
EMBEDDING_MODEL = embedding_models.get("small", os.getenv("DEFAULT_EMBEDDING_MODEL"))

# Create the embedding function
embedding_function = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model=EMBEDDING_MODEL
)


## Function to index documents into ChromaDB
def index_documents(
    documents: list[Document],
    collection_name: str = "knowledge_base",
    persist_directory: str = "db"
):
    """
    Index a list of LangChain Document objects into ChromaDB.

    Args:
        documents: List of LangChain Document objects.
        collection_name: Name of the Chroma collection.
        persist_directory: Path to persist the Chroma DB.
    """
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )
    vectordb.add_documents(documents)
    print(f"✅ Indexed {len(documents)} documents into '{collection_name}'")


def reinitialize_index(
    collection_name: str = "knowledge_base",
    persist_directory: str = "db"
):
    """
    Wipe and recreate the index. Use cautiously.
    """
    import shutil
    shutil.rmtree(persist_directory, ignore_errors=True)
    print("🧹 Previous index removed.")

