# backend/services/text_splitter.py
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_text_splitter(chunk_size: int = 500, chunk_overlap:int = 50):
    """Return a text splitter configured from config."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

def split_documents(docs, chunk_size:int, chunk_overlap:int):
    """Split documents into chunks."""
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    return splitter.split_documents(docs)