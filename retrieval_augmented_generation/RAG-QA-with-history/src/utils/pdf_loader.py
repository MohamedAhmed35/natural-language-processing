"""Small helpers to load uploaded PDF files into LangChain Documents.

This module provides a single helper, ``load_uploaded_pdfs``, which accepts
a list of FastAPI ``UploadFile`` objects, writes each upload to a temporary
PDF file, and uses ``PyMuPDFLoader`` to convert the PDF into a list of
``langchain_core.documents.Document`` objects.
"""

import tempfile
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from fastapi import UploadFile


def load_uploaded_pdfs(uploaded_files: List[UploadFile]):
    """Load uploaded PDF files into LangChain Document objects.

    Args:
        uploaded_files: List of FastAPI ``UploadFile`` instances (usually
            received from a form/file upload endpoint).

    Returns:
        A list of ``langchain_core.documents.Document`` instances produced by
        the ``PyMuPDFLoader`` for each uploaded PDF.

    Side effects:
        - Writes each uploaded file to a temporary file on disk. Temporary
          files are created with ``delete=False`` so they remain accessible
          after closing; they are not removed by this function.
    """
    docs = []
    for f in uploaded_files:
        suffix = ".pdf"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        if tmp:
            content = f.file.read()
            tmp.write(content)
            tmp.flush()
            tmp.close()
            loader = PyMuPDFLoader(tmp.name)
            docs.extend(loader.load())

    return docs
