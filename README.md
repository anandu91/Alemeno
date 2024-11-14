# Document Processing, Vector Store Ingestion, and Query Engine Development using LangChain

Project Overview
This project integrates document processing, vector store ingestion, and query engine development using LangChain, Streamlit, and FAISS. The application allows users to upload a text document, processes the document into manageable chunks, converts the chunks into embeddings, stores them in a vector database (FAISS), and allows querying over the document using an LLM (Language Model) to retrieve relevant information.

Table of Contents
1) Objectives
2) Technologies Used
3) Project Structure
4) Detailed Workflow
5) Results and Evaluation
6) Conclusion

# Objectives
Document Processing: To load and process user-uploaded text documents and split them into smaller chunks for easier handling.
Vector Store Ingestion: To convert the processed document chunks into embeddings using OpenAI’s embeddings API and store them in a FAISS vector database for efficient similarity search.
Query Engine Development: To create an interactive query system that can retrieve relevant chunks of the document based on user queries and generate responses using a language model.
Streamlit Integration: To provide an easy-to-use web interface for uploading documents, querying them, and displaying results interactively.

# Technologies Used
LangChain: A powerful framework for building applications with LLMs, enabling document processing, vectorization, and integration with language models.

langchain.document_loaders: Used to load the documents.
langchain.text_splitter: Used to split documents into chunks.
langchain.embeddings: Used for generating embeddings of document chunks.
langchain.vectorstores: Used for storing and querying document embeddings (FAISS).
langchain.chains: Used to connect LLMs for generating responses.
FAISS (Facebook AI Similarity Search): A vector store that allows for efficient similarity search of embeddings.

OpenAI: For generating embeddings and using a GPT model for querying the document.

Streamlit: A Python framework used to create interactive web applications for data science projects. Used to create a user interface for uploading documents and querying them.

pyngrok: A tool used to tunnel the Streamlit app from a local machine (or Colab) to a publicly accessible URL.

# Project Structure
1. Document Upload and Processing
TextLoader: Used to load text data from the uploaded file.
CharacterTextSplitter: Splits the document into chunks of text for easier processing and efficient querying.
2. Vector Store Creation
OpenAIEmbeddings: Generates embeddings for each document chunk.
FAISS: Stores the document embeddings and provides fast similarity searches.
3. Query Engine Development
LLMChain: Chains a pre-trained language model (like OpenAI’s GPT) to process queries.
PromptTemplate: Defines the input format for the LLM query.
4. Streamlit App
The app allows users to upload .txt files and input queries.
Results show relevant document sections based on similarity search and LLM responses.
