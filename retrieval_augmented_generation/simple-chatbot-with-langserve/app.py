import requests
import streamlit as st


def get_groq_response(input_text, language):
    json_body={
        "input": {
            "language": language,
            "text": input_text
        },
    }  

    response = requests.post("http://127.0.0.1:8000/chain/invoke",
                             json=json_body, 
                             timeout=5).json()
    return response["output"]

# Streamlit app
st.title("Translation App using LLM")
col1, col2 = st.columns([2, 2])  # two equal-width columns
with col1:
    st.markdown("**Enter the text you want to translate to**")
with col2:
    language = st.selectbox(
        "",
        ["English", "German", "French", "Italian", "Portuguese", "Hindi", "Spanish", "Thai"],
        width= 200,
        index=None,
        label_visibility ="collapsed"
    )

input_text = st.text_input("")

if input_text and language:
    translation = get_groq_response(input_text, language)
    st.write("**Translation:**", translation)