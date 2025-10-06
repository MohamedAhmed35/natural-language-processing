# simple-chatbot-with-langserve

Small example project showing how to build a translation app using LangServe, Groq LLM, FastAPI and Streamlit.

## Features
- LangChain-style prompt -> model -> parser chain exposed as an HTTP route using LangServe
- Simple Streamlit front-end to call the chain and display translations
- Minimal configuration via a .env file

## Prerequisites
- Python 3.10+
- pip


## Installation
### 1. Clone
```bash
git clone https://github.com/your-name/natural-language-processing.git
cd retrieval_augmented_generation/simple-chatbot-with-langserve
```

### 2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  (or `.venv\Scripts\activate` on Windows)
```
### 3. Install dependencies:
pip install -r requirements.txt

## Configuration
```bash
cp .env.example .env
```
Set the your values:
- GROQ_API_KEY — your Groq API key (gsk_...)
- LLM_MODEL — model id (e.g., `llama-3.1-8b-instant`)
- APP_NAME, APP_VERSION — optional app metadata


## Running the server
Start the FastAPI / LangServe server:
```bash
python serve.py
```
This exposes the LangServe chain routes at:
http://127.0.0.1:8000/chain/invoke

If running directly with uvicorn:
```bash
uvicorn serve:app --host 127.0.0.1 --port 8000
```

## Running the Streamlit frontend
Start the Streamlit app:
```bash
streamlit run app.py
```

The Streamlit app posts to the chain endpoint (`/chain/invoke`) to request translations and displays the result.

## API Usage
POST /chain/invoke

Body example:
{
  "input": {
    "language": "French",
    "text": "Hello world"
  }
}

Response includes an "output" field with the translated text.

## Project structure (key files)
- serve.py — FastAPI + LangServe routes and chain setup
- app.py — Streamlit UI that calls the chain
- config.py — pydantic settings loader
- .env.example — example environment variables

## Notes
- Ensure the Groq API key and model are valid and have sufficient quota.
- Adjust timeouts and error handling in production.
