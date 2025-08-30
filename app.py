import streamlit as st
import os
import vertexai
from src.helper import download_hugging_face_embeddings, custom_rag_with_fallback
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()


project_id = os.getenv("project_id")
project_region = os.getenv("region")

embeddings = download_hugging_face_embeddings()

vectorstore = Chroma(embedding_function=embeddings, persist_directory="chroma_db")
#vectorstore = build_or_load_chroma(text_chunks, embeddings)

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.1}
)

# authenticate with vertexai
vertexai.init(project=project_id, location=project_region)

#load the model
llm = VertexAI(model_name="gemini-2.5-pro")


prompt_strict = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_strict),
        ("human", "{input}") , # This is where the email text will go
    ]
)

fallback_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_fallback),
        ("human", "{input}"),
    ]
)

# The chain that uses your strict prompt and requires documents
question_answer_chain = create_stuff_documents_chain(llm, prompt_strict)

# The fallback chain that uses the simpler prompt
standalone_llm_chain = fallback_prompt | llm

# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Intelligent Email Assistant", page_icon="✉️")
st.title("📧 Intelligent Email Assistant")

st.write(
    """
    Paste the email you received below. The assistant will summarize it, classify the intent,
    and suggest a professional reply using your knowledge base. If no relevant documents are found,
    a fallback model will generate a response.
    """
)

# Email input box
email_input = st.text_area("Paste your email here:", height=200)

# Submit button
if st.button("Analyze Email"):
    if email_input.strip() == "":
        st.warning("Please paste an email to analyze.")
    else:
        with st.spinner("Analyzing email..."):
            # Call the helper function
            response = custom_rag_with_fallback(
                retriever,
                question_answer_chain,
                standalone_llm_chain,
                email_input
            )
        st.subheader("💡 Suggested Response")
        st.write(response)
