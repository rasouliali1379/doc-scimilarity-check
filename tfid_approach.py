import os
import re
import argparse
from docx import Document
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from hazm import stopwords_list
import ssl


# Temporarily bypass SSL verification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK stopwords if necessary
nltk.download('stopwords')

# Combine English and Persian stop words
english_stopwords = set(nltk.corpus.stopwords.words('english'))
persian_stopwords = set(stopwords_list())
all_stopwords = english_stopwords.union(persian_stopwords)

# Function to extract text from DOCX files
def extract_text_from_docx(file_path):
    """Extracts text from a DOCX file."""
    doc = Document(file_path)
    return ' '.join([para.text for para in doc.paragraphs])

# Function to extract text from PDF files
def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + ' '
        return text

# Function to preprocess text for both English and Persian
def preprocess_text(text):
    """Preprocesses text by lowercasing, removing punctuation, and filtering stop words."""
    # Lowercase the text (affects English only)
    text = text.lower()
    # Replace non-word characters (e.g., punctuation) with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split into words based on spaces
    words = text.split()
    # Remove stop words from both languages
    words = [word for word in words if word not in all_stopwords]
    # Join words back into a single string
    return ' '.join(words)

# Main function to compute document similarity
def check_similarity(existing_docs_dir, new_doc_path):
    """Computes similarity between a new document and existing documents."""
    # Extract and preprocess text from existing documents
    existing_texts = []
    existing_doc_names = []
    for file_name in os.listdir(existing_docs_dir):
        file_path = os.path.join(existing_docs_dir, file_name)
        if file_name.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        elif file_name.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        else:
            continue
        processed_text = preprocess_text(text)
        existing_texts.append(processed_text)
        existing_doc_names.append(file_name)

    # Extract and preprocess text from the new document
    if new_doc_path.endswith('.docx'):
        new_text = extract_text_from_docx(new_doc_path)
    elif new_doc_path.endswith('.pdf'):
        new_text = extract_text_from_pdf(new_doc_path)
    else:
        raise ValueError("Unsupported file format for the new document. Use .docx or .pdf")
    new_text_processed = preprocess_text(new_text)

    # Vectorize texts using TF-IDF
    vectorizer = TfidfVectorizer()
    existing_vectors = vectorizer.fit_transform(existing_texts)
    new_vector = vectorizer.transform([new_text_processed])

    # Compute cosine similarity between the new document and existing documents
    similarity_scores = cosine_similarity(new_vector, existing_vectors)[0]

    # Display similarity scores
    print("\nSimilarity Scores:")
    for doc_name, score in zip(existing_doc_names, similarity_scores):
        print(f"{doc_name}: Similarity = {score:.4f}")

    # Display top 3 most similar documents (if there are at least 3)
    top_n = 3
    if len(similarity_scores) >= top_n:
        sorted_indices = similarity_scores.argsort()[::-1][:top_n]
        print(f"\nTop {top_n} Similar Documents:")
        for idx in sorted_indices:
            print(f"{existing_doc_names[idx]}: Similarity = {similarity_scores[idx]:.4f}")

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check similarity between a new document and existing documents.")
    parser.add_argument('existing_docs_dir', help="Directory containing existing documents (DOCX or PDF)")
    parser.add_argument('new_doc_path', help="Path to the new document (DOCX or PDF)")
    args = parser.parse_args()

    check_similarity(args.existing_docs_dir, args.new_doc_path)