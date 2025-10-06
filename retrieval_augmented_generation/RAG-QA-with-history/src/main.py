# src/main.py
import uvicorn
from fastapi import FastAPI
from api.routes import rag
from api.routes import base

app = FastAPI(title="conversational RAG")

app.include_router(rag.rag_router, prefix="/api/rag")
app.include_router(base.base_router, prefix="/api/rag")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)