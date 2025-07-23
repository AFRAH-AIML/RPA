# 🤖 Local RAG Chatbot: An AI Document Assistant

This project is a powerful, locally-run chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions based on documents you provide. It is designed for privacy, offline functionality, and efficient performance on consumer hardware, all wrapped in a user-friendly web interface built with Streamlit.

---

## 🏩 Architectural Decisions

The core of this application is the Retrieval-Augmented Generation (RAG) architecture. This approach was chosen over using a standard LLM directly because it grounds the AI's responses in factual information from a specific set of documents, preventing hallucinations and ensuring relevance.

The architecture is a two-stage pipeline:

### Retrieval Stage (The "Librarian")

* Converts both the document chunks and the user's query into numerical vectors using a lightweight embedding model.
* A vector database performs a similarity search to retrieve the most relevant context.

### Generation Stage (The "Answer Writer")

* The retrieved context is passed to a generative language model to synthesize and generate a coherent answer to the user's question.

This separation allows use of the best tool for each job: a fast model for searching and a powerful model for answer generation.

---

## 📊 Model Choices

### Generative LLM

**TinyLlama-1.1B-Chat-v1.0.Q4\_K\_M.gguf**

* Chosen for its ability to run offline and its performance on consumer-grade hardware.
* Offers a balance of speed and comprehension.
* Provides sub-15-second response times on CPUs and even faster with GPU offloading.

### Embedding Model

**all-MiniLM-L6-v2**

* Lightweight and fast.
* Standard for high-quality semantic search.
* Automatically downloaded and cached by `sentence-transformers`.

---

## 📜 Chunking and Retrieval Strategy

### Chunking Strategy

* **Document Parsing**: Parsed page-by-page.
* **Sentence Splitting**: Avoids cutting sentences.
* **Chunk Aggregation**: Target size of 512 characters.
* **Overlap**: 50-character overlap to preserve context across chunks.

### Retrieval Approach

* **Vector Database**: Qdrant, running in-memory.
* **Semantic Search**: Uses all-MiniLM-L6-v2 to embed queries.
* **Similarity Metric**: Cosine Similarity for closest matches.
* **Top-K Retrieval**: Retrieves top 3 relevant chunks for LLM input.

---

## 💻 Hardware Usage and Performance

* **Primary bottleneck**: LLM generation step.
* **GPU Offloading**: Enabled using `n_gpu_layers=-1` for max performance.
* **CPU Optimization**: Uses all available CPU threads with `n_threads`.
* **Performance**:

  * LLaMA-2-7B took 40+ seconds on CPU.
  * TinyLlama-1.1B brought it down to 3-15 seconds.
  * Prompt structure highly affects output quality.

---

## 🚀 Setup and Usage

### 1. Prerequisites

* Python 3.8+
* Git

### 2. Setup

```bash
# Clone the repository (replace with your actual URL)
git clone https://github.com/AFRAH-AIML/RPA.git
cd RPA

# Create and activate virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download the Local LLM

* Create a `models` folder in your root directory.
* Download **TinyLlama-1.1B-Chat-v1.0.Q4\_K\_M.gguf** from Hugging Face.
* Place the `.gguf` file inside the `models` folder.

### 4. Run the Application

```bash
streamlit run app.py
```

Your default browser will open with the chatbot interface.

### 5. How to Use

* Use the "Browse files" button to upload `.pdf`, `.docx`, or `.txt` documents.
* Click **Process Documents**.
* Ask questions in the chat box after processing is done.

---

Enjoy your private, local AI-powered document assistant! 🚀
