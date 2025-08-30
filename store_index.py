from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pathlib import Path
from langchain_community.vectorstores import Chroma

extracted_data = load_pdf_file(data="Data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


def build_or_load_chroma(text_chunks, embeddings, persist_dir="chroma_db"):
    
    #Build or load a Chroma vector store from document chunks.
    
    if Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        print("✅ Loaded existing Chroma DB.")
    else:
        vectorstore = Chroma.from_documents(text_chunks, embeddings, persist_directory=persist_dir)
        vectorstore.persist()
        print("✅ Created and persisted new Chroma DB.")
    return vectorstore

vectorstore = build_or_load_chroma(text_chunks, embeddings)