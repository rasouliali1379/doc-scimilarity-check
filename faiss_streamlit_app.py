import os
import numpy as np
from hazm import sent_tokenize
from transformers import AutoModel, AutoTokenizer
import torch
from docx import Document
import PyPDF2
import logging
from datetime import datetime
import faiss
import streamlit as st

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    filename=f'similarity_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Load pre-trained LaBSE model and tokenizer with caching
@st.cache_resource
def load_model():
    model_name = "setu4993/LaBSE"
    logger.info(f"Loading pre-trained LaBSE model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    logger.info("Model and tokenizer loaded successfully")
    faiss.omp_set_num_threads(1)  # Disable FAISS OpenMP parallelism
    logger.info("FAISS OpenMP parallelism set to 1 thread")
    return model, tokenizer

model, tokenizer = load_model()

# Function to extract text from DOCX files
def extract_text_from_docx(file):
    logger.info(f"Extracting text from DOCX")
    try:
        doc = Document(file)
        text = ' '.join([para.text for para in doc.paragraphs])
        logger.info(f"Text extracted from DOCX, length: {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from DOCX: {str(e)}")
        raise

# Function to extract text from PDF files
def extract_text_from_pdf(file):
    logger.info(f"Extracting text from PDF")
    try:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num, page in enumerate(reader.pages, 1):
            extracted = page.extract_text()
            if extracted:
                text += extracted + ' '
            else:
                logger.warning(f"No text extracted from page {page_num}")
        logger.info(f"Text extracted from PDF, length: {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {str(e)}")
        raise

# Function to extract text based on file type
def extract_text(file, file_name):
    if file_name.endswith('.docx'):
        return extract_text_from_docx(file)
    elif file_name.endswith('.pdf'):
        return extract_text_from_pdf(file)
    else:
        logger.error(f"Unsupported file format: {file_name}")
        raise ValueError("Unsupported file format. Use .docx or .pdf")

# Compute embeddings
def compute_embeddings(sentences, model, tokenizer, batch_size=32):
    logger.info(f"Computing embeddings for {len(sentences)} sentences")
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.pooler_output.cpu().numpy()
        embeddings.append(batch_embeddings)
    embeddings = np.vstack(embeddings)
    logger.info(f"Embeddings computed, shape: {embeddings.shape}")
    return embeddings

# Build FAISS index
def build_faiss_index(existing_docs_dir, model, tokenizer, index_path="faiss_index.bin"):
    logger.info(f"Building FAISS index from documents in {existing_docs_dir}")
    
    all_sentences = []
    doc_mapping = []
    
    for file_name in os.listdir(existing_docs_dir):
        file_path = os.path.join(existing_docs_dir, file_name)
        if not (file_name.endswith('.docx') or file_name.endswith('.pdf')):
            continue
        with open(file_path, 'rb') as file:
            text = extract_text(file, file_name)
        sentences = sent_tokenize(text)
        if sentences:
            start_idx = len(all_sentences)
            all_sentences.extend(sentences)
            doc_mapping.append((file_name, int(start_idx), int(len(all_sentences))))
    
    if not all_sentences:
        logger.error("No sentences found in existing documents")
        raise ValueError("No sentences found in existing documents")
    
    embeddings = compute_embeddings(all_sentences, model, tokenizer)
    faiss.normalize_L2(embeddings)
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    logger.info("Adding embeddings to FAISS index")
    index.add(embeddings)
    
    faiss.write_index(index, index_path)
    np.save("doc_mapping.npy", doc_mapping, allow_pickle=True)
    logger.info(f"FAISS index built with {index.ntotal} vectors and saved to {index_path}")
    return index, doc_mapping

# Check similarity with FAISS
def check_similarity_with_faiss(existing_docs_dir, new_doc, new_doc_name, model, tokenizer, index_path="faiss_index.bin"):
    logger.info(f"Starting similarity check with FAISS")
    
    if not os.path.exists(index_path) or not os.path.exists("doc_mapping.npy"):
        logger.info("FAISS index not found, building new index")
        index, doc_mapping = build_faiss_index(existing_docs_dir, model, tokenizer, index_path)
    else:
        index = faiss.read_index(index_path)
        doc_mapping = np.load("doc_mapping.npy", allow_pickle=True).tolist()
        doc_mapping = [(name, int(start), int(end)) for name, start, end in doc_mapping]
        logger.info(f"Loaded FAISS index with {index.ntotal} vectors and document mapping")
    
    new_text = extract_text(new_doc, new_doc_name)
    if not new_text.strip():
        logger.error(f"No text extracted from new document")
        raise ValueError("No text could be extracted from the new document")
    new_sentences = sent_tokenize(new_text)
    if not new_sentences:
        logger.error(f"No sentences detected in new document")
        raise ValueError("No sentences detected in the new document")
    logger.info(f"New document tokenized into {len(new_sentences)} sentences")
    
    new_embeddings = compute_embeddings(new_sentences, model, tokenizer)
    faiss.normalize_L2(new_embeddings)
    
    k = 5
    logger.info(f"Querying FAISS index with k={k}")
    distances, indices = index.search(new_embeddings, k)
    
    doc_similarities = {}
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        for d, j in zip(dist, idx):
            similarity = d
            for doc_name, start_idx, end_idx in doc_mapping:
                if int(start_idx) <= int(j) < int(end_idx):
                    if doc_name not in doc_similarities:
                        doc_similarities[doc_name] = []
                    doc_similarities[doc_name].append(similarity)
                    break
    
    results = []
    for doc_name, sim_scores in doc_similarities.items():
        avg_sim = np.mean(sim_scores)
        prop_high_sim = np.sum(np.array(sim_scores) > 0.8) / len(new_sentences)
        results.append((doc_name, avg_sim, prop_high_sim))
        logger.info(f"{doc_name}: Avg similarity = {avg_sim:.4f}, Prop high similarity = {prop_high_sim:.4f}")
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# Streamlit GUI
def main():
    st.title("Document Similarity Checker")
    st.markdown("Upload a new document and compare it against a directory of existing documents to detect similarities using FAISS and LaBSE embeddings.")

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    existing_docs_dir = st.sidebar.text_input("Existing Documents Directory", value="./test_docs/")
    uploaded_file = st.sidebar.file_uploader("Upload New Document (DOCX or PDF)", type=["docx", "pdf"])

    if st.sidebar.button("Check Similarity"):
        if not os.path.isdir(existing_docs_dir):
            st.error("Please provide a valid directory containing existing documents.")
        elif not uploaded_file:
            st.error("Please upload a new document to compare.")
        else:
            try:
                with st.spinner("Processing..."):
                    results = check_similarity_with_faiss(existing_docs_dir, uploaded_file, uploaded_file.name, model, tokenizer)
                
                st.success("Similarity check completed!")
                st.header("Results (Sorted by Average Similarity)")
                
                for file_name, avg_sim, prop_high_sim in results:
                    with st.expander(f"Existing Document: {file_name}"):
                        st.write(f"**Average Similarity**: {avg_sim:.4f}")
                        st.write(f"**Proportion of Highly Similar Sentences (threshold > 0.8)**: {prop_high_sim:.4f}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Error during similarity check: {str(e)}")

if __name__ == "__main__":
    main()