import os
import tempfile
import google.generativeai as genai
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Response:

    def __init__(self, content):
        self.content = content


st.title("IntelliAssist AI: Smart Document AI Assistant")

# API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# --- SIDEBAR: Document Upload & Processing ---
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                docs = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    loader = PyPDFLoader(tmp_path)
                    docs.extend(loader.load())
                    os.remove(tmp_path)

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                splits = text_splitter.split_documents(docs)

                # Local embedding model (No API dependency/No 404 Error)
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )

                vectorstore = Chroma.from_documents(
                    documents=splits, embedding=embeddings
                )

                st.session_state.retriever = vectorstore.as_retriever()
                st.success("Documents processed successfully!")
        else:
            st.error("Please upload at least one PDF file.")

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.retriever is None:
        with st.chat_message("assistant"):
            st.warning("Please upload and process documents first.")
    else:
        with st.chat_message("assistant"):
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

            # Fetch all supported models dynamically
            available_models = [
                m.name
                for m in genai.list_models()
                if "generateContent" in m.supported_generation_methods
            ]

            # Preference order to avoid deprecated models like 2.5-flash
            preferred_model = None
            for model_candidate in available_models:
                if "3.6-flash" in model_candidate:
                    preferred_model = model_candidate
                    break
                elif "1.5-flash" in model_candidate:
                    preferred_model = model_candidate
                    break

            if not preferred_model:
                preferred_model = available_models[0]

            model = genai.GenerativeModel(preferred_model)

            docs = st.session_state.retriever.invoke(prompt)
            context = "\n\n".join([doc.page_content for doc in docs])
            full_prompt = (
                f"Answer based on context.\nContext: {context}\nQuestion: {prompt}"
            )

            response_obj = model.generate_content(full_prompt)

            response = Response(response_obj.text)
            sources = set(
                [doc.metadata.get("source", "Unknown") for doc in docs]
            )
            source_text = "\n\n**Sources:**\n" + "\n".join(
                [f"- {src}" for src in sources]
            )

            full_response = response.content + source_text
            st.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )