import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings import OpenAIEmbeddings  # Example local embeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp  # Assuming LLaMA model is available locally
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Assuming you have LLaMA model installed locally (via LlamaCpp)
llama_model = LlamaCpp(model_path="path_to_llama_model.bin")  # Path to LLaMA model binary file

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    try:
        embeddings = OpenAIEmbeddings()  # Local embeddings, replace with suitable local embedding method
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error generating vector store: {e}")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "answer is not available in the context," don't provide the wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = LLMChain(llm=llama_model, prompt=prompt)
    return chain

def extract_key_information(text):
    # Function to extract specific information such as revenue or risk factors
    risk_factors = "risk factors: (some regex or keyword extraction logic)"
    revenue_info = "total revenue for Google Search: (some regex or keyword extraction logic)"
    
    return risk_factors, revenue_info

def compare_documents(documents, query):
    # Compare documents based on the query
    # Use FAISS and embeddings to find similar sections from different documents
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local("faiss_index", embeddings)
        docs = vector_store.similarity_search(query)
        
        chain = get_conversational_chain()
        response = chain.run({"context": docs, "question": query})
        
        return response
    except Exception as e:
        st.error(f"Error during document comparison: {e}")

def user_input(user_question):
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local("faiss_index", embeddings)
        docs = vector_store.similarity_search(user_question)

        # Check for specific questions related to risk factors or revenue
        if "risk factors" in user_question.lower():
            # Extract risk factors from the docs (simple search or regex)
            risk_factors, _ = extract_key_information(docs)
            response = f"Risk factors identified: {risk_factors}"
        elif "revenue" in user_question.lower():
            # Extract revenue-related information
            _, revenue_info = extract_key_information(docs)
            response = f"Revenue information: {revenue_info}"
        else:
            # Default response via the conversational chain
            chain = get_conversational_chain()
            response = chain.run({"context": docs, "question": user_question})

        st.write("Reply: ", response)
    except Exception as e:
        st.error(f"Error in user input processing: {e}")

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Local LLaMA Model💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                except Exception as e:
                    st.error(f"Error during PDF processing: {e}")

if __name__ == "__main__":
    main()

