import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

# Load the Groq API key
groq_api_key = os.environ["GROQ_API_KEY"]

# Initialize session state for vector storage
if "vectors" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://timesofindia.indiatimes.com/city/dehradun/case-against-man-posing-as-jay-shah-to-extort-rs-3-cr-from-rudrapur-mla/articleshow/118336993.cms")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    st.session_state.final_document = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_document, st.session_state.embeddings)

st.title("ChatGroq Model")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template(
    "Answer the question based on the context only.\n"
    "Please provide the most accurate response based on the question.\n\n"
    "<context>\n{context}\n</context>\n\n"
    "Question: {input}"
)

document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Input field
prompt = st.text_input("Input your prompt here")

if prompt:
    start = time.process_time()
    
    # Retrieve relevant documents first
    retrieved_docs = retriever.invoke(prompt)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Ensure both 'context' and 'input' are passed
    response = retrieval_chain.invoke({"input": prompt, "context": context})

    st.write("Response Time:", time.process_time() - start)

    if "answer" in response:
        st.write(response["answer"])
    else:
        st.write("No response generated.")

    # Show the retrieved documents
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(retrieved_docs):
            st.write(doc.page_content)
            st.write("-----------------------")
