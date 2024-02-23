import time

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils import add_doc_to_db

with open("../.openaikey/api-key.ini", "r") as f:
    OPENAI_KEY = f.read()

MODEL_NAME = "text-embedding-3-small"
embeddings = OpenAIEmbeddings(
    model=MODEL_NAME,
    api_key=OPENAI_KEY,
)
vectordb = FAISS.load_local(
    folder_path="/Users/pujaris/Documents/lablab/data/neuroscience/faiss_index_hp",
    embeddings=embeddings,
)
llm_model = ChatOpenAI(
    # model="gpt-3.5-turbo",
    model="gpt-4-turbo-preview",
    temperature=0,
    openai_api_key=OPENAI_KEY,
)
template = """
Answer the question in the Language captured in the question,
Answer the question based only on the following context: 
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

## RAG Chain
rag_chain = (
    {
        "context": vectordb.as_retriever(
            search_kwargs={"k": 3, "search_type": "similarity"}
        ),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm_model
    | StrOutputParser()
)


def main():
    st.title(":books: Books made easy")
    # Create two columns for input/output and PDF upload
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("Query")
        input_text = st.text_area("Enter Text", height=300)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                output_text = rag_chain.invoke(input_text)
                st.header("Output")
                st.write(
                    output_text,
                    height=300,
                )
    with col2:
        st.header("Upload PDF")
        uploaded_file = st.file_uploader(
            "Upload",
            type=["pdf"],
        )
        if uploaded_file:
            with st.spinner("Processing..."):
                add_doc_to_db(uploaded_file)
                st.success("Processing finished.")


if __name__ == "__main__":
    main()







