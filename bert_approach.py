import os
import numpy as np
from hazm import sent_tokenize
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from docx import Document
import PyPDF2
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename=f'similarity_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Load LaBSE model and tokenizer
model_name = "setu4993/LaBSE"
logger.info(f"Loading LaBSE model and tokenizer: {model_name}")
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
logger.info("Model and tokenizer loaded successfully")

# Function to extract text from DOCX files
def extract_text_from_docx(file_path):
    """Extracts text from a DOCX file."""
    logger.info(f"Extracting text from DOCX: {file_path}")
    try:
        doc = Document(file_path)
        text = ' '.join([para.text for para in doc.paragraphs])
        logger.info(f"Text extracted from {file_path}, length: {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from {file_path}: {str(e)}")
        raise

# Function to extract text from PDF files
def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    logger.info(f"Extracting text from PDF: {file_path}")
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page_num, page in enumerate(reader.pages, 1):
                extracted = page.extract_text()
                if extracted:
                    text += extracted + ' '
                else:
                    logger.warning(f"No text extracted from page {page_num} in {file_path}")
            logger.info(f"Text extracted from {file_path}, length: {len(text)} characters")
            return text
    except Exception as e:
        logger.error(f"Failed to extract text from {file_path}: {str(e)}")
        raise

# Function to extract text based on file type
def extract_text(file_path):
    """Extracts text from a file based on its extension."""
    if file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    else:
        logger.error(f"Unsupported file format: {file_path}")
        raise ValueError("Unsupported file format. Use .docx or .pdf")

# Function to compute sentence embeddings using LaBSE
def compute_embeddings(sentences):
    """
    Compute LaBSE embeddings for a list of sentences.
    
    Args:
        sentences (list of str): List of sentences.
    
    Returns:
        numpy.ndarray: Embeddings for the sentences.
    """
    logger.info(f"Computing embeddings for {len(sentences)} sentences")
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.pooler_output
    logger.info(f"Embeddings computed, shape: {embeddings.shape}")
    return embeddings.cpu().numpy()

# Function to compute similarity metrics between new sentences and existing sentences
def compute_similarity_metrics(new_sentences, existing_sentences):
    """
    Compute similarity metrics between new sentences and existing sentences.
    
    Args:
        new_sentences (list of str): Sentences from the new document.
        existing_sentences (list of str): Sentences from an existing document.
    
    Returns:
        tuple: (average similarity, proportion of highly similar sentences)
    """
    logger.info(f"Computing similarity metrics: {len(new_sentences)} new sentences vs {len(existing_sentences)} existing sentences")
    new_embeddings = compute_embeddings(new_sentences)
    existing_embeddings = compute_embeddings(existing_sentences)
    
    max_similarities = []
    for i, new_emb in enumerate(new_embeddings):
        similarities = cosine_similarity([new_emb], existing_embeddings)[0]
        max_similarity = np.max(similarities)
        max_similarities.append(max_similarity)
        logger.debug(f"Sentence {i+1} max similarity: {max_similarity:.4f}")
    
    average_similarity = np.mean(max_similarities)
    proportion_high_similarity = np.sum(np.array(max_similarities) > 0.8) / len(max_similarities)
    logger.info(f"Average similarity: {average_similarity:.4f}, Proportion highly similar: {proportion_high_similarity:.4f}")
    return average_similarity, proportion_high_similarity

# Main function to check similarity between new document and existing documents
def check_similarity(existing_docs_dir, new_doc_path):
    """
    Check similarity between a new document and existing documents.
    
    Args:
        existing_docs_dir (str): Directory containing existing documents (DOCX or PDF).
        new_doc_path (str): Path to the new document (DOCX or PDF).
    """
    logger.info(f"Starting similarity check: New doc = {new_doc_path}, Existing docs dir = {existing_docs_dir}")
    
    # Extract and tokenize the new document
    new_text = extract_text(new_doc_path)
    if not new_text.strip():
        logger.error(f"No text extracted from {new_doc_path}")
        raise ValueError(f"No text could be extracted from {new_doc_path}")
    new_sentences = sent_tokenize(new_text)
    if not new_sentences:
        logger.error(f"No sentences detected in {new_doc_path}")
        raise ValueError(f"No sentences detected in {new_doc_path}")
    logger.info(f"New document tokenized into {len(new_sentences)} sentences")

    # Process each existing document
    results = []
    for file_name in os.listdir(existing_docs_dir):
        file_path = os.path.join(existing_docs_dir, file_name)
        if not (file_name.endswith('.docx') or file_name.endswith('.pdf')):
            logger.warning(f"Skipping {file_path}: Unsupported format")
            continue
        
        logger.info(f"Processing existing document: {file_name}")
        existing_text = extract_text(file_path)
        if not existing_text.strip():
            logger.warning(f"No text extracted from {file_name}, skipping...")
            continue
        
        existing_sentences = sent_tokenize(existing_text)
        if not existing_sentences:
            logger.warning(f"No sentences detected in {file_name}, skipping...")
            continue
        logger.info(f"{file_name} tokenized into {len(existing_sentences)} sentences")
        
        # Compute similarity metrics
        avg_sim, prop_high_sim = compute_similarity_metrics(new_sentences, existing_sentences)
        results.append((file_name, avg_sim, prop_high_sim))
    
    if not results:
        logger.error("No valid existing documents found to compare.")
        print("No valid existing documents found to compare.")
        return
    
    # Sort results by average similarity (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Output and log the results
    print("Similarity Results (sorted by average similarity):")
    logger.info("Similarity Results (sorted by average similarity):")
    for file_name, avg_sim, prop_high_sim in results:
        result_str = (
            f"Existing Document: {file_name}\n"
            f"  Average Similarity: {avg_sim:.4f}\n"
            f"  Proportion of Highly Similar Sentences (threshold > 0.8): {prop_high_sim:.4f}\n"
        )
        print(result_str)
        logger.info(result_str.strip())

    logger.info("Similarity check completed")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check similarity between a new Persian document and existing documents with logging.")
    parser.add_argument('existing_docs_dir', help="Directory containing existing Persian documents (DOCX or PDF)")
    parser.add_argument('new_doc_path', help="Path to the new Persian document (DOCX or PDF)")
    args = parser.parse_args()

    check_similarity(args.existing_docs_dir, args.new_doc_path)