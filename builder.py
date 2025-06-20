# builder.py
import os
import json
import hashlib
import argparse
import logging
import shutil
import time
import fitz
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- Logger for GUI Communication ---


def setup_builder_logger():
    logger = logging.getLogger('Builder')
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler('builder.log', 'w', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    return logger


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
            # For the adapter
            'unprocessed_knowledge': os.path.join(self.paths['cache'], 'unprocessed_knowledge.json'),
            'embeddings': os.path.join(self.paths['cache'], 'embeddings.npy'),
            'final_index': os.path.join(self.paths['kb'], 'faiss_index.bin'),
            'final_meta': os.path.join(self.paths['kb'], 'documents_meta.json')
        }

        if self.args.force_rebuild:
            self.wipe_cache()

        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)

    def wipe_cache(self):
        logger.info("LOG|WARNING|Force rebuild requested. Wiping cache...")
        shutil.rmtree(self.paths['cache'], ignore_errors=True)
        os.makedirs(self.paths['cache'], exist_ok=True)

    def get_file_hash(self, filepath):
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()

    def infer_metadata(self, filename):
        fn_lower = filename.lower()
        if 'handbook' in fn_lower or 'manual' in fn_lower:
            return {'type': 'clinical_handbook', 'authority': 0.95}
        if 'harrison' in fn_lower or 'cecil' in fn_lower:
            return {'type': 'reference_tome', 'authority': 1.0}
        if 'davidson' in fn_lower or 'kumar' in fn_lower:
            return {'type': 'student_textbook', 'authority': 0.9}
        return {'type': 'general_textbook', 'authority': 0.7}

    def process_pdf_worker(self, filepath):
        try:
            filename = os.path.basename(filepath)
            chunks = []
            metadata = self.infer_metadata(filename)
            doc = fitz.open(filepath)
            for page_num, page in enumerate(doc):
                text = " ".join(page.get_text("text").split())
                if len(text) < 100:
                    continue

                words = text.split()
                chunk_size, overlap = 300, 50
                for i in range(0, len(words), chunk_size - overlap):
                    chunk_text = " ".join(words[i:i + chunk_size])
                    if len(chunk_text) > 50:
                        chunks.append({
                            "text": chunk_text,
                            "source_file": filename, "page": page_num + 1, **metadata
                        })
            return (filepath, chunks, None)
        except Exception as e:
            return (filepath, [], str(e))

    def run(self):
        logger.info("LOG|INFO|==== Starting Knowledge Base Sync ====")
        try:
            # 1. Analyze for changed files
            try:
                with open(self.artifacts['cache_index'], 'r') as f:
                    cache_index = json.load(f)
            except:
                cache_index = {}

            files_to_process = [
                os.path.join(self.paths['library'], f) for f in os.listdir(self.paths['library'])
                if f.endswith('.pdf') and (f not in cache_index or cache_index[f] != self.get_file_hash(os.path.join(self.paths['library'], f)))
            ]

            if not files_to_process:
                logger.info(
                    "LOG|SUCCESS|Knowledge Base is already up-to-date. No new documents to process.")
                return True

            logger.info(
                f"LOG|INFO|Found {len(files_to_process)} new or updated documents to process.")

            # 2. Chunk new/updated PDFs in parallel
            with Pool(processes=min(cpu_count(), len(files_to_process))) as pool:
                results = list(tqdm(pool.imap(self.process_pdf_worker, files_to_process), total=len(
                    files_to_process), desc="Processing PDFs"))

            # 3. Update the permanent chunk list and the adapter's queue
            try:
                with open(self.artifacts['chunks'], 'r') as f:
                    all_chunks = json.load(f)
            except:
                all_chunks = []

            try:
                with open(self.artifacts['unprocessed_knowledge'], 'r') as f:
                    unprocessed_knowledge = json.load(f)
            except:
                unprocessed_knowledge = []

            processed_filenames = [os.path.basename(r[0]) for r in results]
            # Filter out old chunks from files that are being updated
            all_chunks = [c for c in all_chunks if c['source_file']
                          not in processed_filenames]

            newly_generated_chunks_count = 0
            for filepath, new_chunks, error in results:
                if error:
                    logger.info(
                        f"LOG|ERROR|Failed to process {os.path.basename(filepath)}: {error}")
                else:
                    all_chunks.extend(new_chunks)  # For RAG index
                    unprocessed_knowledge.extend(
                        new_chunks)  # For the AI to adapt to
                    newly_generated_chunks_count += len(new_chunks)

            with open(self.artifacts['chunks'], 'w') as f:
                json.dump(all_chunks, f)
            with open(self.artifacts['unprocessed_knowledge'], 'w') as f:
                json.dump(unprocessed_knowledge, f)
            logger.info(
                f"LOG|SUCCESS|Queued {newly_generated_chunks_count} new text chunks for AI adaptation.")

            # 4. Update the hash cache
            for fp in files_to_process:
                cache_index[os.path.basename(fp)] = self.get_file_hash(fp)
            with open(self.artifacts['cache_index'], 'w') as f:
                json.dump(cache_index, f, indent=2)

            logger.info("LOG|SUCCESS|Knowledge Base Sync complete.")
            return True

        except Exception as e:
            logger.info(
                f"LOG|ERROR|A fatal error occurred in the builder: {e}", exc_info=True)
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Base Builder for MedAI")
    parser.add_argument('--force-rebuild', action='store_true',
                        help="Force full rebuild of index (deletes cache)")
    args = parser.parse_args()

    engine = BuildEngine(args)
    success = engine.run()


if __name__ == "__main__":
    logger = setup_builder_logger()
    main()
