import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv("utils/.env")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE")) 
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def load_pdf_chunks(path, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    loader = PyPDFLoader(path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(documents)
    for c in chunks:
        c.metadata["source"] = os.path.basename(path)
    return chunks

