## Server using LangServe
import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from fastapi import FastAPI
from langserve import add_routes
from .config import settings


load_dotenv()

model = ChatGroq(model=settings.LLM_MODEL, groq_api_key=settings.GROQ_API_KEY)


# Create prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ('system', "Translate the following into {language}"),
    ('human', "{text}")
])

output_parser = StrOutputParser()


# Create chain
chain = prompt_template|model|output_parser

# App definition
app = FastAPI(title="LangChain Server",
              version=settings.APP_VERSION,
              description="A simple API server using Langchain runnable interfaces")

# Adding chain routes
add_routes(
    app,
    chain,
    path="/chain"
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)