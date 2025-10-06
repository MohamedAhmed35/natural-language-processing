# src/routes/rag.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from services.vectorstore import VectorStoreManager
from services.rag_chain import build_chain
from services.text_splitter import split_documents
from utils.pdf_loader import load_uploaded_pdfs
from api.schemas import ChatRequest, Response
from utils.logger import get_logger


rag_router = APIRouter(tags=["rag"])
logger = get_logger(__name__)
vector_mgr = VectorStoreManager()



@rag_router.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload PDFs.
    """
    try:
        docs = load_uploaded_pdfs(files)
        texts = split_documents(docs, chunk_size = 2000, chunk_overlap=100)
        # add documents
        vector_mgr.add_documents(texts)
        if vector_mgr.count():
            return {"status": "ok", "message": "Uploaded and indexed PDFs."}
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(e))
    

@rag_router.post("/chat")
async def chat(request: ChatRequest):
    """
    Query RAG with session-aware history
    """
    chain = build_chain(vector_mgr)
    try:
        response = chain.invoke(
            {
                "input": request.question
            },
            config = {
                "configurable": {
                    "session_id": request.session_id
                }
            }
        )
        
        response_payload = Response(
            status = "ok",
            answer = response["answer"],
            context = response["context"]
        )
        return response_payload.model_dump()
        
    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(e))
    

@rag_router.post("/reset_session")
async def reset_session(session_id: str):
    """
    Clear conversation history for a session id.
    """
    from src.services.rag_chain import CHAT_HISTORIES
    if session_id in CHAT_HISTORIES:
        CHAT_HISTORIES.pop(session_id)
    
    return {
        "status": "ok",
        "message": f"Cleared history for {session_id}"
    }
