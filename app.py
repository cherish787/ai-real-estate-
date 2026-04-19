import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from PIL import Image
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv()

def extract_text_from_file(file):
    """Extract text from uploaded PDF or text file"""
    text = ""
    
    if file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file.type == "text/plain":
        text = file.read().decode("utf-8")
    
    return text

def split_text_into_chunks(text):
    """Split text into manageable chunks for vector store"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    """Create FAISS vector store from text chunks using Gemini embeddings"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def get_answer_from_llm(vector_store, user_question):
    """Retrieve relevant documents and get answer from LLM"""
    if not vector_store:
        return "Please upload documents first before asking questions."
    
    docs = vector_store.similarity_search(user_question, k=3)
    
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.3)
    
    prompt_template = """
    You are an AI Real Estate Copilot. Answer the question based only on the provided context.
    If you don't find the answer in the context, say "I cannot find information about this in the uploaded documents."
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
    response = chain.invoke({"input_documents": docs, "question": user_question})
    
    return response["output_text"]

def main():
    st.set_page_config(page_title="AI Real Estate Copilot", page_icon="🏠", layout="wide")
    st.title("🏠 AI Real Estate Copilot")
    
    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'all_text' not in st.session_state:
        st.session_state.all_text = ""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    # Sidebar for file uploads
    with st.sidebar:
        st.subheader("📁 Upload Documents")
        
        # Image uploads
        uploaded_images = st.file_uploader(
            "Upload Images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True
        )
        
        # Document uploads
        uploaded_docs = st.file_uploader(
            "Upload Documents (PDF / TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
        
        process_button = st.button("🔄 Process Documents", type="primary", use_container_width=True)
        
        if process_button and uploaded_docs:
            with st.spinner("Processing documents..."):
                all_text = ""
                
                for doc in uploaded_docs:
                    text = extract_text_from_file(doc)
                    all_text += text + "\n\n"
                
                st.session_state.all_text = all_text
                
                if all_text.strip():
                    chunks = split_text_into_chunks(all_text)
                    st.session_state.vector_store = create_vector_store(chunks)
                    st.success(f"✅ Processed {len(uploaded_docs)} documents, {len(chunks)} chunks created")
                else:
                    st.warning("No text could be extracted from the uploaded files")
    
    # Display uploaded images
    if uploaded_images:
        st.subheader("📷 Uploaded Images")
        cols = st.columns(3)
        for idx, img_file in enumerate(uploaded_images):
            with cols[idx % 3]:
                img = Image.open(img_file)
                st.image(img, caption=img_file.name, use_column_width=True)
        st.divider()
    
    # Question answering section
    st.subheader("💬 Ask Questions About Your Documents")
    
    user_question = st.text_input("Enter your question:")
    
    if user_question:
        with st.spinner("Searching documents and generating answer..."):
            answer = get_answer_from_llm(st.session_state.vector_store, user_question)
            
            st.info("### Answer")
            st.write(answer)
            
            # Show sources
            if st.session_state.vector_store:
                with st.expander("View Relevant Document Sections"):
                    sources = st.session_state.vector_store.similarity_search(user_question, k=3)
                    for i, source in enumerate(sources):
                        st.write(f"**Source {i+1}:**")
                        st.write(source.page_content)
                        st.divider()
    
    # Display extracted text
    if st.session_state.all_text:
        with st.expander("View All Extracted Text"):
            st.text_area("", st.session_state.all_text, height=300)

if __name__ == "__main__":
    main()