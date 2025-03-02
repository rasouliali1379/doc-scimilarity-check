import os
import numpy as np
from hazm import sent_tokenize
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
import torch
from torch import nn
from docx import Document
import PyPDF2
import logging
from datetime import datetime
import random
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from datasets import Dataset

# Disable tokenizers parallelism to avoid forking issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    filename=f'finetune_similarity_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# --- Fine-Tuning Section ---

# Load LaBSE model and tokenizer
model_name = "setu4993/LaBSE"
logger.info(f"Loading pre-trained LaBSE model and tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
logger.info("Model and tokenizer loaded successfully")

# Function to extract text from DOCX files
def extract_text_from_docx(file_path):
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
    if file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    else:
        logger.error(f"Unsupported file format: {file_path}")
        raise ValueError("Unsupported file format. Use .docx or .pdf")

# Generate training pairs (positive and negative)
def generate_training_pairs(docs_dir):
    logger.info(f"Generating training pairs from documents in {docs_dir}")
    all_sentences = []
    doc_sentences = []
    
    for file_name in os.listdir(docs_dir):
        file_path = os.path.join(docs_dir, file_name)
        if not (file_name.endswith('.docx') or file_name.endswith('.pdf')):
            continue
        text = extract_text(file_path)
        sentences = sent_tokenize(text)
        if sentences:
            all_sentences.extend(sentences)
            doc_sentences.append(sentences)
    
    logger.info(f"Total sentences extracted: {len(all_sentences)}")
    
    pairs = []
    labels = []
    
    # Positive pairs: consecutive sentences from the same document
    for sentences in doc_sentences:
        for i in range(len(sentences) - 1):
            pairs.append((sentences[i], sentences[i + 1]))
            labels.append(1)
    
    # Negative pairs: random sentences from different documents
    num_negative = len(labels)
    for _ in range(num_negative):
        sent1 = random.choice(all_sentences)
        sent2 = random.choice(all_sentences)
        if sent1 != sent2 and abs(len(sent1) - len(sent2)) > 10:
            pairs.append((sent1, sent2))
            labels.append(0)
    
    logger.info(f"Generated {len(pairs)} pairs: {len([l for l in labels if l == 1])} positive, {num_negative} negative")
    return pairs, labels

# Prepare dataset for training
def prepare_dataset(pairs, labels, tokenizer, max_length=512):
    logger.info("Preparing dataset for training")
    # Tokenize sentence pairs separately
    encodings = tokenizer(
        [pair[0] for pair in pairs],  # Sentence 1
        [pair[1] for pair in pairs],  # Sentence 2
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Convert to list of dictionaries for Dataset
    dataset_dict = {
        "input_ids": encodings["input_ids"].tolist(),
        "attention_mask": encodings["attention_mask"].tolist(),
        "labels": labels
    }
    dataset = Dataset.from_dict(dataset_dict)
    logger.info(f"Dataset prepared with {len(dataset)} examples")
    return dataset

# Custom model for contrastive loss
class ContrastiveModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.pooler_output
        
        # Split embeddings into sentence1 and sentence2
        batch_size = input_ids.size(0) // 2
        sent1_emb = embeddings[0::2]  # First sentence of each pair
        sent2_emb = embeddings[1::2]  # Second sentence of each pair
        
        # Cosine similarity
        cos_sim = nn.functional.cosine_similarity(sent1_emb, sent2_emb)
        
        if labels is not None:
            # Ensure labels match the batch size after pairing
            labels = labels[0::2]  # Take every other label to match cos_sim size
            loss = nn.MSELoss()(cos_sim, labels.float())
            return {"loss": loss}
        return {"cos_sim": cos_sim}

# Fine-tune the model
def fine_tune_model(docs_dir, output_dir="fine_tuned_labse"):
    logger.info("Starting fine-tuning process")
    
    # Generate training data
    pairs, labels = generate_training_pairs(docs_dir)
    dataset = prepare_dataset(pairs, labels, tokenizer)
    
    # Initialize the custom model
    contrastive_model = ContrastiveModel(model)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_steps=10,
        save_steps=100,
        learning_rate=2e-5,
        report_to="none"
    )
    
    # Trainer
    trainer = Trainer(
        model=contrastive_model,
        args=training_args,
        train_dataset=dataset
    )
    
    logger.info("Training started")
    trainer.train()
    logger.info("Training completed")
    
    # Save the fine-tuned model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Fine-tuned model saved to {output_dir}")

# --- Similarity Checking Section ---

# Function to compute sentence embeddings with the fine-tuned model
def compute_embeddings(sentences, model, tokenizer):
    logger.info(f"Computing embeddings for {len(sentences)} sentences")
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.pooler_output
    logger.info(f"Embeddings computed, shape: {embeddings.shape}")
    return embeddings.cpu().numpy()

# Function to compute similarity metrics
def compute_similarity_metrics(new_sentences, existing_sentences, model, tokenizer):
    logger.info(f"Computing similarity metrics: {len(new_sentences)} new vs {len(existing_sentences)} existing sentences")
    new_embeddings = compute_embeddings(new_sentences, model, tokenizer)
    existing_embeddings = compute_embeddings(existing_sentences, model, tokenizer)
    
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

# Check similarity with the fine-tuned model
def check_similarity(existing_docs_dir, new_doc_path, model_dir="fine_tuned_labse"):
    logger.info(f"Starting similarity check with fine-tuned model from {model_dir}")
    
    # Load fine-tuned model
    fine_tuned_model = AutoModel.from_pretrained(model_dir)
    fine_tuned_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    logger.info("Fine-tuned model and tokenizer loaded")
    
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

    # Process existing documents
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
        avg_sim, prop_high_sim = compute_similarity_metrics(new_sentences, existing_sentences, fine_tuned_model, fine_tuned_tokenizer)
        results.append((file_name, avg_sim, prop_high_sim))
    
    if not results:
        logger.error("No valid existing documents found to compare.")
        print("No valid existing documents found to compare.")
        return
    
    # Sort results by average similarity
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Output and log results
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

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LaBSE and check similarity for Persian documents.")
    parser.add_argument('mode', choices=['finetune', 'check'], help="Mode: 'finetune' to fine-tune model, 'check' to check similarity")
    parser.add_argument('existing_docs_dir', help="Directory containing existing Persian documents (DOCX or PDF)")
    parser.add_argument('--new_doc_path', help="Path to the new Persian document (DOCX or PDF, required for 'check' mode)")
    parser.add_argument('--output_dir', default="fine_tuned_labse", help="Directory to save the fine-tuned model (default: fine_tuned_labse)")
    
    args = parser.parse_args()
    
    if args.mode == 'finetune':
        fine_tune_model(args.existing_docs_dir, args.output_dir)
    elif args.mode == 'check':
        if not args.new_doc_path:
            parser.error("'check' mode requires --new_doc_path")
        check_similarity(args.existing_docs_dir, args.new_doc_path, args.output_dir)