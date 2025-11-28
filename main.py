import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from io import StringIO


# LLM loader
def load_LLM(openai_api_key):
    return ChatOpenAI(
        model="gpt-4o-mini",    # modelo más económico compatible
        temperature=0,
        api_key=openai_api_key
    )


# Page config
st.set_page_config(page_title="AI Long Text Summarizer")
st.header("AI Long Text Summarizer")


# Intro
col1, col2 = st.columns(2)
with col1:
    st.markdown("Summarize long documents with this app.")
with col2:
    st.write("Contact with [AI Accelera](https://aiaccelera.com)")


# API Key input
st.markdown("## Enter Your OpenAI API Key")
openai_api_key = st.text_input(
    label="OpenAI API Key",
    placeholder="sk-...",
    type="password"
)


# File uploader
st.markdown("## Upload the text file you want to summarize")
uploaded_file = st.file_uploader("Choose a file", type="txt")


# Output title
st.markdown("### Here is your Summary:")


# Logic
if uploaded_file is not None:
    text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()

    # Word limit
    if len(text.split()) > 20000:
        st.write("Please upload a file below 20,000 words.")
        st.stop()

    # Key required
    if not openai_api_key:
        st.warning(
            "Please insert your OpenAI API Key.",
            icon="⚠️"
        )
        st.stop()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=5000,
        chunk_overlap=350
    )
    docs = splitter.create_documents([text])

    # Load model
    llm = load_LLM(openai_api_key)

    # Summarization chain
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce"
    )

    # New API → use invoke, not run()
    result = chain.invoke({"input_documents": docs})

    # Result is in "output_text"
    st.write(result["output_text"])
