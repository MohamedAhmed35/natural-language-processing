"""
Streamlit front-end for the RAG (Retrieval-Augmented Generation) Q&A demo.

This module provides a minimal Streamlit UI that lets a user:
- upload one or more PDF documents to be indexed by the backend
- ask questions about the uploaded documents and view the generated answer
"""

import os
from uuid import uuid4
import requests
import streamlit as st
from typing import List
from src.config import settings


# Base API URL for the RAG backend (configured in src.config.settings)
API_URL = settings.RAG_API_URL


# --- Streamlit UI setup -------------------------------------------------
st.title("RAG Q&A with history")

# We store it in Streamlit's session_state so it
# survives reruns while the browser session is active.
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{uuid4().hex[:8]}"


# --- File upload / indexing ---------------------------------------------
# Allow the user to upload multiple PDF files. Files are compared by filename
# against the previously uploaded list in session_state to avoid re-uploading
# the same files repeatedly during the same session.
uploaded_files = st.file_uploader(
    "Upload PDF(s) to index", type="pdf", accept_multiple_files=True
)

# Keep the previously uploaded files (if any) so we can detect newly added PDFs
previous_upload = st.session_state.get("previous_upload", [])
query = None

if uploaded_files:
    # Extract names for compare. 
    uploaded_names = [f.name for f in uploaded_files]
    prev_names = [f.name for f in previous_upload]

    # Determine which files are newly added compared to the previous upload
    new_pdfs = [f for f in uploaded_files if f.name not in prev_names]
    if new_pdfs:
        # Prepare multipart file tuples: (fieldname, (filename, bytes, content_type)) 
        files = [("files", (f.name, f.getvalue(), f.type)) for f in new_pdfs]

        # POST files to the backend /upload endpoint. 
        response = requests.post(f"{API_URL}/upload", files=files)
        if response.ok:
            st.success("Uploaded and indexed.")
            # Save the upload list so duplicate uploads are avoided in this session
            st.session_state.previous_upload = uploaded_files
        else:
            # Surface backend error details to the user
            st.error(response.text)

    st.markdown("---")
    query = st.text_input("Ask a question about the uploaded documents:")



# --- Chat / Q&A interaction ---------------------------------------------
if query:
    # Build the JSON payload expected by the backend chat endpoint.
    payload = {
        "session_id": st.session_state.session_id,
        "question": query,
    }

    # Send the question to the backend; backend returns the answer and the
    # retrieved context (list of document chunks) used to produce that answer.
    response = requests.post(f"{API_URL}/chat", json=payload)
    if response.ok:
        data = response.json()
        st.markdown("**Answer:**")
        # Write the main answer text. Use .get with a default to avoid KeyError
        st.write(data.get("answer", ""))

        # Provide the retrieved context in an expander so users can inspect
        # the document passages that informed the model's answer.
        with st.expander("Retrieved context"):
            for doc in data.get("context", []):
                st.write(doc.get("page_content", ""))
                st.write("---")
    else:
        # Show backend error messages to help debugging (simple UX for demo)
        st.error(response.text)

