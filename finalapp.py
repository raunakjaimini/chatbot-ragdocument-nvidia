import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import time
import tempfile

# Load the Groq API key
load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Set up the page
st.set_page_config(page_title="Chat-Mate Nvidia Demo", page_icon="📄", layout="centered")
st.title("Chat-Mate pdf reader using Nvidia NIM ")
st.write("Upload a PDF document and ask questions based on the content.")

# Function to handle vector embedding
def vector_embedding(pdf_file):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path)  # Use the path to the temporary file
        docs = loader.load()  # Document Loading
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunk Creation
        final_documents = text_splitter.split_documents(docs)  # Splitting
        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)  # Vector Store

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

# Check if the user has uploaded a PDF
if uploaded_file:
    st.success("PDF Uploaded Successfully!")

    if st.button("Process Document for Embedding"):
        with st.spinner("Processing..."):
            vector_embedding(uploaded_file)
        st.success("Vector Store DB Is Ready")

    # User input
    prompt1 = st.text_input("Enter Your Question Based on the Document")

    # Create and process the retrieval chain
    if prompt1:
        llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question.
            <context>
            {context}
            <context>
            Question: {input}
            """
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        end = time.process_time()
        
        st.write("### Answer:")
        st.write(response['answer'])

        st.write(f"⏱️ Response time: {end - start:.2f} seconds")

        # Document similarity search
        with st.expander("🔍 Document Similarity Search"):
            st.write("Relevant Document Chunks:")
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("--------------------------------")

# Styling
st.markdown("""
    <style>
        .css-1d391kg, .css-12ttj6m {
            background-color: #2c2f33;
            color: #ffffff;
        }
        .stButton>button {
            color: white;
            background-color: #7289da;
            border-radius: 8px;
        }
        .stTextInput>div>input {
            background-color: #23272a;
            color: white;
            border-radius: 8px;
            padding: 10px;
        }
        .stFileUploader>div>div>div>button {
            background-color: #7289da;
            color: white;
            border-radius: 8px;
        }
        .stExpander>div>div>div {
            background-color: #2c2f33;
            border-radius: 8px;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)
