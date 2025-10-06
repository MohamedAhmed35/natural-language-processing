"""RAG chain builder and in-memory session history manager.

This module provides a single high-level function, ``build_chain``, which
creates and returns a history-aware retrieval-augmented generation (RAG)
pipeline wrapped as a ``RunnableWithMessageHistory``.
  - Construct a question-rewriting contextualizer to convert conversational
      queries into self-contained questions using the chat history.
  - Create a history-aware retriever that uses the provided vectorstore
      manager to fetch context relevant to the (rewritten) question.
  - Compose a concise QA chain that answers only from retrieved context.
  - Trim the chat history to remain under the model's token limits before
      each request.
"""
from typing import Any, Dict

from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory

from config import settings
from services.vectorstore import VectorStoreManager



# In-memory chat histories keyed by session_id
CHAT_HISTORIES = {}


def _get_session_history(session_id: str):
    if session_id not in CHAT_HISTORIES:
        CHAT_HISTORIES[session_id] = ChatMessageHistory()

    return CHAT_HISTORIES[session_id]


def build_chain(vectorstore_mgr: VectorStoreManager):
    """
    Create a retrieval + QA chain with history-aware retriever and a trimmer.
    Returns a RunnableWithMessageHistory ready for .invoke(...)
    """
    llm = ChatGroq(groq_api_key=settings.GROQ_API_KEY, model_name=settings.LLM_MODEL)

    # History-aware contextualizer
    contextualize_q_system_prompt = (
        "You are a question rewriter. "
        "Given the chat history and the latest user question, rewrite the question so it is fully self-contained. "
        "Include any missing context or references from the chat history needed to make it clear and specific. "
        "If the question already makes sense alone, return it unchanged.\n\n"
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    retriever = vectorstore_mgr.as_retriever()
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # QA prompt expects {context}
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context only to answer "
        "the question. If you cannot answer from the retrieved context, say that I "
        "don't know. Use three to five sentences maximum and keep the "
        "answer concise.\n\ncontex:\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt),
          MessagesPlaceholder("chat_history"),
         ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Trimmer runnable to keep messages under token limit
    trimmer = RunnableLambda(
        lambda x: {
            **x,
            "chat_history": trim_messages(
                x.get("chat_history", []),
                token_counter=llm,
                max_tokens=settings.TRIM_MAX_TOKENS,
                strategy="last"
            ),
        }
    )

    chain = trimmer | rag_chain

    # Wrap with message history accessor
    runnable_with_history = RunnableWithMessageHistory(
        chain,
        _get_session_history,
        input_message_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return runnable_with_history