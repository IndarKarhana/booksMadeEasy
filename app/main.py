import time
from tempfile import NamedTemporaryFile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

template = """
Make sure to answer the question in english unless the question is in different Language,
Answer the question based only on the following context, add the file-name and page-numbers at the end of the answer for reference.: 
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    st.title("📚 Books made easy")
    # two columns for input/output and PDF upload
    col1, col2 = st.columns([2, 1])

    with col2:
        st.header("Enter OpenAi-key")
        OPENAI_KEY = st.text_input("Password:", type="password")

        st.header("Upload PDF")
        uploaded_file = st.file_uploader(
            "Upload",
            type=["pdf"],
        )

        if uploaded_file and OPENAI_KEY:
            with st.spinner("Processing..."):
                # NamedTemporaryFile to handle the uploaded file
                with NamedTemporaryFile(delete=False) as temp_file:
                    # content of the uploaded file to the temporary file
                    temp_file.write(uploaded_file.read())
                    temp_file.flush()

                    loader = PyPDFLoader(temp_file.name)
                    docs = loader.load()
                
                # Attempt to create an instance of ChatOpenAI to verify the key
                try:
                    llm_model = ChatOpenAI(
                        model="gpt-4-turbo-preview",
                        temperature=0,
                        openai_api_key=OPENAI_KEY,
                    )
                except Exception as e:
                    st.error(f"Invalid API Key: {e}")
                    return
                
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    api_key=OPENAI_KEY,
                )
                vectordb = FAISS.from_documents(
                    documents=docs,
                    embedding=embeddings,
                )
                st.success("Processing finished.")
                
                # RAG Chain setup
                rag_chain_from_docs = (
                    RunnablePassthrough.assign(
                        context=(lambda x: format_docs(x["context"]))
                    )
                    | prompt
                    | llm_model
                    | StrOutputParser()
                )

                rag_chain_with_source = RunnableParallel(
                    {
                        "context": vectordb.as_retriever(
                            search_kwargs={"k": 3, "search_type": "similarity"}
                        ),
                        "question": RunnablePassthrough(),
                    }
                ).assign(answer=rag_chain_from_docs)

                with col1:
                    st.header("Query")
                    input_text = st.text_area("Enter Text", height=300)
                    if st.button("Submit"):
                        with st.spinner("Processing..."):
                            response = rag_chain_with_source.invoke(input_text)
                            output_text = response["answer"]

                            output_text += (
                                f" Following pages from {uploaded_file.name} has additional information for reference: "
                                + str([doc.metadata["page"] for doc in response["context"]])
                            )
                            st.header("Output")
                            st.write(output_text)
        elif not OPENAI_KEY:
            st.error("Please enter an API key.")
    st.sidebar.image("./booksmadeeasy.png", caption='', use_column_width=True)     
    st.sidebar.markdown("### Disclaimer")
    st.sidebar.write(
        "Tier 1 OpenAI key is expected. The key is not saved or stored anywhere. \
            It is only used for authentication purposes during the session"
    )


if __name__ == "__main__":
    main()
