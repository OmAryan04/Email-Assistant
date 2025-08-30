from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents

#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks



def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings



def custom_rag_with_fallback(retriever, question_answer_chain, standalone_llm_chain, email_text):
    """
    Handles the email processing with a fallback to a standalone LLM if no
    documents are found by the retriever.
    """
    try:
        # First, run the retriever to find documents based on the email content.
        retrieved_docs = retriever.invoke(email_text)
        
        # Check if any documents were returned.
        if retrieved_docs:
            print("Found relevant documents. Using the RAG chain.")
            # If documents exist, pass them to your RAG chain.
            response = question_answer_chain.invoke({"input": email_text, "context": retrieved_docs})
            return response
        else:
            print("No documents found. Falling back to the standalone LLM.")
            # If no documents were found, use the standalone LLM chain.
            response = standalone_llm_chain.invoke({"input": email_text})
            return response
            
    except Exception as e:
        # This will catch and print any error that happens during the LLM call.
        print(f"An error occurred during the LLM call: {e}")
        # Return an empty string or a custom error message
        return "Sorry, I am unable to process this request at the moment due to an error."