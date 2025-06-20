# builder.py (Version 2 - Now creates train/eval splits)
import os
import fitz
import json
import random
from tqdm import tqdm

print("--- MedAI Knowledge Packet Builder (V2) ---")

PDF_SOURCE_PATH = '../data/library/'
DATASET_PATH = '../data_for_colab/'  # We'll put datasets for the trainer here


def chunk_text(text):
    words = text.split()
    chunk_size, overlap = 300, 50
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_text = " ".join(words[i:i + chunk_size])
        if len(chunk_text.split()) > 20:
            chunks.append(chunk_text)
    return chunks


def build_datasets_for_colab(eval_split_ratio=0.05):
    if not os.path.exists(PDF_SOURCE_PATH) or not os.listdir(PDF_SOURCE_PATH):
        print(f"Error: The directory '{PDF_SOURCE_PATH}' is empty.")
        return

    os.makedirs(DATASET_PATH, exist_ok=True)
    all_chunks_with_meta = []
    pdf_files = [f for f in os.listdir(
        PDF_SOURCE_PATH) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF(s) to process.")

    for filename in tqdm(pdf_files, desc="Processing PDFs"):
        filepath = os.path.join(PDF_SOURCE_PATH, filename)
        doc = fitz.open(filepath)
        for page_num, page in enumerate(doc):
            text = " ".join(page.get_text("text").split())
            if not text:
                continue

            page_chunks = chunk_text(text)
            for chunk_text in page_chunks:
                # We'll use a text formatting that the trainer will expect
                formatted_text = f"Analyze the following medical text and understand its implications. Text: {chunk_text}"
                all_chunks_with_meta.append({"text": formatted_text})

    if not all_chunks_with_meta:
        print("Error: No text extracted.")
        return

    print(f"Total chunks created: {len(all_chunks_with_meta)}")
    random.shuffle(all_chunks_with_meta)  # Shuffle before splitting

    split_index = int(len(all_chunks_with_meta) * (1 - eval_split_ratio))
    train_data = all_chunks_with_meta[:split_index]
    eval_data = all_chunks_with_meta[split_index:]

    with open(os.path.join(DATASET_PATH, 'train_dataset.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)

    with open(os.path.join(DATASET_PATH, 'eval_dataset.json'), 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2)

    print("\n--- Success! ---")
    print(f"Datasets for Colab created in: {DATASET_PATH}")
    print(f"Training set size: {len(train_data)} chunks")
    print(f"Evaluation set size: {len(eval_data)} chunks")


if __name__ == "__main__":
    build_datasets_for_colab()
