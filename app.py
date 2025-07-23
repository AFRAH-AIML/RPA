import streamlit as st
# --- CRITICAL CHANGE: Import Llama for GGUF models ---
from llama_cpp import Llama
from process_data import DocumentProcessor, create_and_store_embeddings, search_collection
from qdrant_client import QdrantClient
import time
import gc
import os # Imported os to get CPU count

# --- Configuration ---
# Using a unique collection name to avoid conflicts.
COLLECTION_NAME = "document_chunks_local_llama_v1"
# --- SPEED OPTIMIZATION: Switched to a smaller, faster model ---
# You will need to download this model file.
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MAX_RESPONSE_TIME = 180 # Increased timeout for local model inference.

# --- Session State Initialization ---
if 'processor' not in st.session_state:
    st.session_state.processor = DocumentProcessor()
if 'qdrant_client' not in st.session_state:
    st.session_state.qdrant_client = QdrantClient(":memory:")
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Core AI and App Functions ---

@st.cache_resource
def load_llm():
    """
    Load the GGUF model from the local file path using llama-cpp-python.
    """
    with st.spinner(f"Loading local AI model from {MODEL_PATH}..."):
        try:
            # --- PERFORMANCE OPTIMIZATION 1: Use optimal number of CPU threads ---
            # Set n_threads to the number of physical CPU cores you have.
            # os.cpu_count() gives logical cores, so dividing by 2 is a good estimate for physical cores.
            # Adjust this number based on your specific hardware for best performance.
            n_threads = os.cpu_count() // 2 if os.cpu_count() else 4

            model = Llama(
                model_path=MODEL_PATH,
                n_ctx=2048,      # Context window size
                n_gpu_layers=-1, # Offload all possible layers to GPU (-1 for all)
                n_threads=n_threads, # Set the number of CPU threads
                verbose=False    # Set to True for detailed logging
            )
            return model
        except Exception as e:
            st.error(f"Fatal Error: Failed to load the local AI model. Please ensure the path is correct and the file is not corrupted. Details: {e}")
            return None

def generate_real_answer(model: Llama, context: str, question: str, source_info: list) -> str:
    """
    This function uses a simplified and more direct prompt format to guide the
    smaller model more effectively, preventing confused or circular answers.
    """
    # --- PROMPT OPTIMIZATION: Using a more direct prompt format ---
    # This format is clearer for smaller models and reduces confusion.
    system_prompt = "You are an expert assistant. Your task is to answer the user's question based *only* on the provided context. Do not use any outside knowledge."
    user_prompt = f"Context: \"{context}\"\n\nQuestion: \"{question}\""

    # Using the specific chat template for TinyLlama for better results.
    full_prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>\n"


    try:
        # Using the raw completion endpoint for more control over the prompt.
        response = model(
            prompt=full_prompt,
            temperature=0.2, # Lower temperature for more factual, less creative answers
            max_tokens=150,
            stop=["<|end|>", "Question:", "Context:"], # Stop generation at these tokens
            echo=False # Don't repeat the prompt in the output
        )
        
        answer = response['choices'][0]['text'].strip()

        # Check for empty or refusal answers from the model.
        if not answer or "don't know" in answer.lower() or "cannot find" in answer.lower() or "not contain" in answer.lower():
             answer = "I found relevant information in the documents, but could not construct a specific answer. The key information is on the pages listed below."

        # Append the source documents for transparency.
        if source_info:
            sources_text = "\n\n**Sources:**\n"
            processed_sources = set()
            for i, source in enumerate(source_info[:3], 1): # Show top 3 sources
                source_id = f"{source['filename']} (Page: {source['page_num']})"
                if source_id not in processed_sources:
                    sources_text += f"{i}. {source_id}\n"
                    processed_sources.add(source_id)
            # Add sources only if the answer is not the fallback message
            if "could not construct" not in answer:
                 answer += sources_text
            else:
                 answer += sources_text.replace("**Sources:**", "")


        return answer

    except Exception as e:
        return f"An error occurred while the AI was generating the response: {e}"


def main():
    """
    The main function that runs the Streamlit application.
    """
    st.set_page_config(page_title="Local Llama RAG Chatbot", page_icon="🤖", layout="wide")
    
    st.title("🤖 Document Chatbot with Local Llama-2")
    st.markdown("Upload documents and ask questions to get answers from your own local AI!")

    model = load_llm()
    if model is None:
        return

    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Choose PDF, DOCX, or TXT files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Analyzing and indexing documents..."):
                all_chunks = []
                for file in uploaded_files:
                    document = st.session_state.processor.load_document(file)
                    chunks = st.session_state.processor.create_chunks_with_metadata(document)
                    all_chunks.extend(chunks)
                    st.success(f"✅ Analyzed {file.name}")
                
                create_and_store_embeddings(
                    st.session_state.qdrant_client,
                    st.session_state.processor,
                    all_chunks,
                    COLLECTION_NAME
                )
                st.session_state.documents_processed = True
                st.success("🎉 All documents are indexed and ready!")
        
        if st.session_state.documents_processed:
            st.success("✅ Ready for queries!")

    if not st.session_state.documents_processed:
        st.info("👆 Please upload and process your documents to begin.")
        return

    st.header("💬 Ask Your Questions")
    
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    if query := st.chat_input("Ask a question about your documents..."):
        st.session_state.chat_history.append(("user", query))
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and formulating an answer..."):
                start_time = time.time()
                
                search_results = search_collection(
                    st.session_state.qdrant_client,
                    st.session_state.processor,
                    query,
                    COLLECTION_NAME,
                    top_k=3
                )
                
                if not search_results:
                    response = "I couldn't find any relevant information in the documents to answer that question."
                else:
                    context = "\n\n".join([result['text'] for result in search_results])
                    response = generate_real_answer(model, context, query, search_results)
                
                response_time = time.time() - start_time
                response += f"\n\n*Response generated in {response_time:.2f} seconds.*"
                
                st.markdown(response)
                st.session_state.chat_history.append(("assistant", response))

    gc.collect()

if __name__ == "__main__":
    main()
