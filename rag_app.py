import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

# Load environment variables
load_dotenv()

# Initialize session state
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None

def create_rag_system(uploaded_file, api_key):
    # Set the API key
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Save uploaded file temporarily
    with open("temp_data.txt", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Load document
    loader = TextLoader("temp_data.txt")
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = GooglePalmEmbeddings()
    vectorstore = Chroma.from_documents(texts, embeddings)
    
    # Create Gemini model instance
    llm = GoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.7,
        google_api_key=api_key
    )

    
    # Create retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    # Clean up temp file
    os.remove("temp_data.txt")
    
    return qa_chain

def main():
    st.title("📚 RAG Question-Answering System with Gemini")
    
    # API Key handling
    api_key = st.text_input("Enter your Google API key:", type="password")
    if not api_key:
        st.warning("Please enter your Google API key to proceed")
        return
    
    # File upload section
    st.subheader("1. Upload Your Document")
    uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing document..."):
            st.session_state.qa_system = create_rag_system(uploaded_file, api_key)
        st.success("Document processed! You can now ask questions.")
    
    # Question-answering section
    st.subheader("2. Ask Questions")
    question = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        if st.session_state.qa_system is None:
            st.warning("Please upload and process a document first!")
        elif not question:
            st.warning("Please enter a question!")
        else:
            with st.spinner("Thinking..."):
                answer = st.session_state.qa_system.run(question)
                st.write("Answer:", answer)

if __name__ == "__main__":
    main()
