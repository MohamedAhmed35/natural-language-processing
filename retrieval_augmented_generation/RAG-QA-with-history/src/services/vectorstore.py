"""Vector store manager backed by Chroma.

This module provides a thin wrapper, ``VectorStoreManager``, around a
Chroma vector store instance to centralize initialization, persistence
directory handling, and common helper operations used by the RAG pipeline.
  - Create and initialize the Chroma store with a chosen embedding model.
  - Provide a convenience ``as_retriever`` method tuned for MMR search.
  - Add documents and report an approximate document count.
"""

import os
from typing import List, Optional
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import settings
from utils.logger import get_logger


logger = get_logger(__name__)


class VectorStoreManager:
    """Encapsulate a persistent Chroma vector store.

    This class centralizes setup and common operations. It intentionally
    keeps a small surface area: initialization, adding documents,
    returning a retriever, and a simple count method.

    Public methods
    - add_documents(docs: List[Document]) -> None
      Persist a batch of Document objects to the vector store. No-op on
      empty input.

    - as_retriever(search_kwargs: Optional[dict]) -> Retriever
      Return a configured retriever instance using MMR search. Accepts
      optional ``search_kwargs`` passed to Chroma's retriever factory.

    - count() -> int
      Return the number of stored document ids, or 0 on error.
    """

    def __init__(self, embedding_model: Optional[str] = None):
        # Use provided model name or fall back to project settings
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        # Persist directory is located under PROJECT_ROOT/persist_dir
        self.persist_dir = os.path.join(settings.PROJECT_ROOT, "persist_dir")
        self.store = None
        self._init_store()

    def _init_store(self):
        """Initialize the Chroma store and embeddings if not already set.

        Currently uses OllamaEmbeddings with a fixed model identifier.
        Swap in a different embedding provider here if desired.
        """
        if self.store is None:
            # embeddings = HuggingFaceEmbeddings(model_name = self.embedding_model)
            embeddings = OllamaEmbeddings(model="embeddinggemma:latest")
            self.store = Chroma(
                embedding_function=embeddings,
                persist_directory=self.persist_dir
            )
            logger.info("Initialized Chroma vectorstore at %s", self.persist_dir)

    def add_documents(self, docs: List[Document]):
        """Add a list of LangChain Document objects to the vectorstore.

        Args:
            docs: List of ``langchain_core.documents.Document`` instances.

        Behavior:
            - If ``docs`` is empty or falsy, the method is a no-op.
            - Delegates to Chroma's ``add_documents`` method for persistence.
        """
        if not docs:
            return
        self.store.add_documents(docs)

    def as_retriever(self, search_kwargs=None):
        """Return a retriever configured for MMR search.

        The default search kwargs are tuned to return a compact set of
        candidates (k=3) with MMR diversity. Pass ``search_kwargs`` to
        override defaults when needed.
        """
        return self.store.as_retriever(search_type='mmr', search_kwargs=search_kwargs or {"k": 3, "lambda_mult": 0.4, "fetch_k": 10})
    
    def count(self):
        """Return the number of stored documents.

        Returns 0 if the store is not yet initialized or on error.
        """
        try:
            data = self.store.get()
            return len(data.get("ids", []))
        except Exception:
            return 0
        