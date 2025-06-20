# builder.py
import os
import json
import hashlib
import argparse
import logging
import shutil
import time
import fitz
import faiss
import numpy as np
import torch
from multiprocessing import Pool, cpu_count
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure logging with cleaner format
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Simpler format for GUI integration
    filename='builder.log',
    filemode='w'
)
logger = logging.getLogger(__name__)


class BuildEngine:
    def __init__(self, args):
        self.args = args
        self.paths = {
            'library': 'data/library',
            'cache': 'cache',
            'kb': 'knowledge_base'
        }
        self.artifacts = {
            'cache_index': os.path.join(self.paths['cache'], 'cache_index.json'),
            'chunks': os.path.join(self.paths['cache'], 'all_chunks.json'),
            'embeddings': os.path.join(self.paths['cache'], 'embeddings.npy'),
            'final_index': os.path.join(self.paths['kb'], 'faiss_index.bin'),
            'final_meta': os.path.join(self.paths['kb'], 'documents_meta.json')
        }

        if self.args.force_rebuild:
            self.wipe_cache()

        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)

    def wipe_cache(self):
        logger.info("Force rebuild requested. Wiping cache...")
        shutil.rmtree(self.paths['cache'], ignore_errors=True)
        os.makedirs(self.paths['cache'], exist_ok=True)

    def get_file_hash(self, filepath):
        """Compute MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def infer_metadata(self, filename):
        """Infer metadata based on filename patterns"""
        fn_lower = filename.lower()
        if 'handbook' in fn_lower or 'manual' in fn_lower:
            return {'type': 'clinical_handbook', 'authority': 0.95}
        if 'harrison' in fn_lower or 'cecil' in fn_lower:
            return {'type': 'reference_tome', 'authority': 1.0}
        if 'davidson' in fn_lower or 'kumar' in fn_lower:
            return {'type': 'student_textbook', 'authority': 0.9}
        return {'type': 'general_textbook', 'authority': 0.7}

    def process_pdf(self, filepath):
        """Process a PDF file into text chunks"""
        try:
            filename = os.path.basename(filepath)
            chunks = []
            doc = fitz.open(filepath)

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text").strip()
                if not text:
                    continue

                # Split into words and create chunks
                words = text.split()
                chunk_size = 300
                overlap = 50

                for i in range(0, len(words), chunk_size - overlap):
                    chunk_text = " ".join(words[i:i + chunk_size])
                    if len(chunk_text) > 50:
                        chunks.append({
                            "text": chunk_text,
                            "source_file": filename,
                            "page": page_num + 1,
                            **self.infer_metadata(filename)
                        })
            return (filepath, chunks, None)
        except Exception as e:
            return (filepath, [], str(e))

    def analyze_library_changes(self):
        """Identify new or modified PDF files - this is the synchronization part"""
        logger.info("Analyzing library for changes...")
        try:
            with open(self.artifacts['cache_index'], 'r') as f:
                cache_index = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            cache_index = {}

        changed_files = []
        for filename in os.listdir(self.paths['library']):
            if not filename.endswith('.pdf'):
                continue

            filepath = os.path.join(self.paths['library'], filename)
            current_hash = self.get_file_hash(filepath)

            # If file not in cache or hash changed, it needs processing
            if filename not in cache_index or cache_index[filename] != current_hash:
                changed_files.append(filepath)

        return changed_files

    def stage_1_chunk(self, files_to_process):
        """Process PDFs into text chunks - incremental processing"""
        if not files_to_process:
            logger.info("No files need processing. Using existing chunks.")
            return True

        logger.info(f"Processing {len(files_to_process)} new/updated files...")
        start_time = time.time()

        # Process files in parallel
        num_workers = min(cpu_count(), len(files_to_process))
        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(self.process_pdf, files_to_process),
                                total=len(files_to_process),
                                desc="Processing PDFs"))

        # Load existing chunks
        try:
            with open(self.artifacts['chunks'], 'r') as f:
                all_chunks = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_chunks = []

        # Remove old versions of modified files
        processed_filenames = [os.path.basename(r[0]) for r in results]
        all_chunks = [c for c in all_chunks
                      if c['source_file'] not in processed_filenames]

        # Add new chunks
        for filepath, chunks, error in results:
            filename = os.path.basename(filepath)
            if error:
                logger.error(f"Failed to process {filename}: {error}")
            else:
                all_chunks.extend(chunks)
                logger.info(f"Added {len(chunks)} chunks from {filename}")

        # Save updated chunks
        with open(self.artifacts['chunks'], 'w') as f:
            json.dump(all_chunks, f, indent=2)
        logger.info(f"Total chunks: {len(all_chunks)}")

        # Update cache index
        for filepath in files_to_process:
            filename = os.path.basename(filepath)
            cache_index[filename] = self.get_file_hash(filepath)

        with open(self.artifacts['cache_index'], 'w') as f:
            json.dump(cache_index, f, indent=2)

        logger.info(f"Chunking completed in {time.time()-start_time:.2f}s")
        return True

    def stage_2_embed(self):
        """Generate embeddings for text chunks"""
        logger.info("Generating embeddings...")
        start_time = time.time()

        # Load chunks
        try:
            with open(self.artifacts['chunks'], 'r') as f:
                chunks = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.error("No chunks found. Please run chunking stage first.")
            return False

        if not chunks:
            logger.error("No chunks to embed. Aborting embedding stage.")
            return False

        # Setup model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

        # Determine optimal batch size based on VRAM
        def get_batch_size(vram_gb):
            if vram_gb >= 10:
                return 256
            if vram_gb >= 6:
                return 128
            if vram_gb >= 4:
                return 64
            return 32

        batch_size = 32
        if device == 'cuda':
            vram_gb = torch.cuda.get_device_properties(
                0).total_memory / (1024**3)
            batch_size = get_batch_size(vram_gb)
            logger.info(
                f"GPU detected: {vram_gb:.1f}GB VRAM. Using batch size: {batch_size}")

        # Generate embeddings
        texts = [c['text'] for c in chunks]
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Save embeddings
        np.save(self.artifacts['embeddings'], embeddings)
        logger.info(f"Created embeddings: {embeddings.shape}")
        logger.info(f"Embedding completed in {time.time()-start_time:.2f}s")
        return True

    def stage_3_index(self):
        """Build FAISS index from embeddings"""
        logger.info("Building FAISS index...")
        start_time = time.time()

        # Load embeddings
        try:
            embeddings = np.load(self.artifacts['embeddings'])
        except FileNotFoundError:
            logger.error(
                "No embeddings found. Please run embedding stage first.")
            return False

        # Load chunks for metadata
        try:
            with open(self.artifacts['chunks'], 'r') as f:
                chunks = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.error("No chunks found. Please run chunking stage first.")
            return False

        # Build index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Save index and metadata
        faiss.write_index(index, self.artifacts['final_index'])
        with open(self.artifacts['final_meta'], 'w') as f:
            json.dump(chunks, f, indent=2)

        logger.info(f"Index size: {index.ntotal} vectors")
        logger.info(f"Indexing completed in {time.time()-start_time:.2f}s")
        return True

    def run(self):
        """Execute the full pipeline with incremental updates"""
        logger.info("==== Starting Knowledge Base Sync ====")
        start_time = time.time()
        success = True

        try:
            # 1. Find new/changed files
            changed_files = self.analyze_library_changes()

            # 2. Process only changed files (incremental update)
            if changed_files:
                logger.info(f"Found {len(changed_files)} new/updated files")
                if not self.stage_1_chunk(changed_files):
                    logger.error("Chunking stage failed")
                    success = False
            else:
                logger.info("No new/updated files found")

            # 3. Only run embedding/indexing if we have chunks
            if success and os.path.exists(self.artifacts['chunks']):
                # Only embed if we processed new files or embeddings don't exist
                if changed_files or not os.path.exists(self.artifacts['embeddings']):
                    if not self.stage_2_embed():
                        logger.error("Embedding stage failed")
                        success = False

                # Only index if we have new embeddings or index doesn't exist
                if success and (changed_files or not os.path.exists(self.artifacts['final_index'])):
                    if not self.stage_3_index():
                        logger.error("Indexing stage failed")
                        success = False
            else:
                logger.warning(
                    "Skipping embedding/indexing - no chunks available")
                success = False

            if success:
                logger.info(
                    f"Sync completed successfully in {time.time()-start_time:.2f} seconds")
            else:
                logger.error("Sync completed with errors")

            return success
        except Exception as e:
            logger.exception(f"Fatal error during sync: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Knowledge Base Builder")
    parser.add_argument('--force-rebuild', action='store_true',
                        help="Force full rebuild of index (deletes cache)")
    args = parser.parse_args()

    engine = BuildEngine(args)
    success = engine.run()

    if success:
        print("SYSTEM|SUCCESS|Sync completed successfully")
    else:
        print("SYSTEM|ERROR|Sync completed with errors")


if __name__ == "__main__":
    main()
