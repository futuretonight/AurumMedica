# build_rag_index.py
import os
import sys
import json
import hashlib
import argparse
import time
import logging
import shutil
from multiprocessing import Pool, cpu_count, Manager
from collections import defaultdict
import faiss
import fitz
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from colorama import init, Fore, Style
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import queue

# --- Utility Class for Colorized Logging ---


class ColorLogger:
    def __init__(self, log_queue): self.q = log_queue

    def log(self, level, msg, exc_info=False):
        log_entry = f"[{level.upper()}] {time.strftime('%H:%M:%S')} - {msg}"
        if exc_info:
            import traceback
            log_entry += f"\n{traceback.format_exc()}"
        self.q.put(log_entry)

# --- GUI Log Viewer ---


class LogViewer(tk.Toplevel):
    def __init__(self, master, log_queue):
        super().__init__(master)
        self.title("Phoenix Pipeline - Live Debug Log")
        self.geometry("900x600")
        self.log_queue = log_queue
        self.text = ScrolledText(self, state='disabled', bg='black', fg='white', font=(
            "Consolas", 9), relief="sunken", bd=2)
        self.text.pack(expand=True, fill='both')
        self.text.tag_config('DEBUG', foreground='#888888')
        self.text.tag_config('INFO', foreground='cyan')
        self.text.tag_config('SUCCESS', foreground='lime')
        self.text.tag_config('WARNING', foreground='yellow')
        self.text.tag_config('ERROR', foreground='red',
                             font=("Consolas", 9, "bold"))
        self.protocol("WM_DELETE_WINDOW", master.destroy)
        self.after(100, self.poll_log_queue)

    def poll_log_queue(self):
        while not self.log_queue.empty():
            self.display(self.log_queue.get(block=False))
        self.after(100, self.poll_log_queue)

    def display(self, message):
        self.text.configure(state='normal')
        self.text.insert(tk.END, message + '\n')
        for level in ['SUCCESS', 'DEBUG', 'INFO', 'WARNING', 'ERROR']:
            if f'[{level}]' in message:
                self.text.tag_add(
                    level, f"{self.text.index('end-2l')}", f"{self.text.index('end-1l')}break")
                break
        self.text.configure(state='disabled')
        self.text.yview(tk.END)

# --- Core Processing Functions ---


def get_optimal_batch_size(vram_gb):
    if vram_gb >= 10:
        return 256
    if vram_gb >= 6:
        return 128
    if vram_gb >= 4:
        return 64
    return 32


def infer_metadata(fn):
    fn_lower = fn.lower()
    if 'handbook' in fn_lower or 'manual' in fn_lower:
        return {'type': 'clinical_handbook', 'authority': 0.95}
    if 'harrison' in fn_lower or 'cecil' in fn_lower:
        return {'type': 'reference_tome', 'authority': 1.0}
    if 'davidson' in fn_lower or 'kumar' in fn_lower:
        return {'type': 'student_textbook', 'authority': 0.9}
    return {'type': 'general_textbook', 'authority': 0.7}


def process_pdf_worker(filepath, q):
    try:
        q.put(
            f"[DEBUG] Worker {os.getpid()} starting: {os.path.basename(filepath)}")
        chunks, metadata = [], infer_metadata(os.path.basename(filepath))
        doc = fitz.open(filepath)
        for page_num, page in enumerate(doc):
            text = " ".join(page.get_text("text").split())
            if not text:
                continue
            words, chunk_size, overlap = text.split(), 300, 50
            for i in range(0, len(words), chunk_size - overlap):
                chunk_text = " ".join(words[i:i + chunk_size])
                if len(chunk_text) > 50:
                    chunks.append({"text": chunk_text, "source_file": os.path.basename(
                        filepath), "page": page_num + 1, **metadata})
        q.put(
            f"[INFO] Worker {os.getpid()} finished '{os.path.basename(filepath)}', found {len(chunks)} chunks.")
        return (filepath, chunks)
    except Exception as e:
        q.put(
            f"[ERROR] Worker {os.getpid()} failed on '{os.path.basename(filepath)}': {e}")
        return (filepath, [])

# --- The Main Resilient Pipeline Class ---


class PhoenixPipeline:
    def __init__(self, log_queue, args):
        self.logger = ColorLogger(log_queue)
        self.args = args
        self.paths = {'library': 'data/library',
                      'cache': 'cache', 'kb': 'knowledge_base'}
        self.artifacts = {'cache_index': os.path.join(self.paths['cache'], 'cache_index.json'), 'chunks': os.path.join(self.paths['cache'], 'all_chunks.json'),
                          'embeddings': os.path.join(self.paths['cache'], 'embeddings.npy'), 'final_index': os.path.join(self.paths['kb'], 'faiss_index.bin'),
                          'final_meta': os.path.join(self.paths['kb'], 'documents_meta.json')}
        if self.args.force_rebuild:
            self.wipe_cache()
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)

    def wipe_cache(self): self.logger.log('warning', "Force rebuild requested. Wiping cache..."); shutil.rmtree(
        self.paths['cache'], ignore_errors=True)

    def run(self):
        try:
            self.logger.log(
                'success', "===== Phoenix Pipeline Activated =====")
            new_or_changed_files = self.analyze_library_changes()
            if not new_or_changed_files and os.path.exists(self.artifacts['final_index']):
                self.logger.log(
                    'success', "Library is fully synchronized. No actions needed.")
                return
            self.stage_1_chunk(new_or_changed_files)
            self.stage_2_embed()
            self.stage_3_index()
            self.logger.log(
                'success', "Pipeline complete. Knowledge Base is synchronized and ready.")
        except Exception as e:
            self.logger.log(
                'error', f"A FATAL error occurred in the pipeline", exc_info=True)

    def get_file_hash(self, fp): h = hashlib.md5(); b = bytearray(128*1024); f = open(fp, 'rb',
                                                                                      buffering=0); [h.update(b[:n]) for n in iter(lambda: f.readinto(b), 0)]; return h.hexdigest()

    def analyze_library_changes(self):
        self.logger.log(
            'info', "Analyzing library for new or modified files...")
        try:
            with open(self.artifacts['cache_index'], 'r') as f:
                cache_index = json.load(f)
        except:
            cache_index = {}
        changed_files = [f for f in os.listdir(self.paths['library']) if f.endswith('.pdf') and (
            f not in cache_index or cache_index[f] != self.get_file_hash(os.path.join(self.paths['library'], f)))]
        return changed_files

    def stage_1_chunk(self, files_to_process):
        self.logger.log(
            'info', f"Stage 1 [Chunking]: Found {len(files_to_process)} files to process.")
        if not files_to_process:
            self.logger.log(
                'info', "No new files to chunk. Moving to next stage.")
            return

        filepaths = [os.path.join(self.paths['library'], f)
                     for f in files_to_process]
        num_workers = min(cpu_count(), len(filepaths))
        with Pool(processes=num_workers) as pool, Manager() as manager:
            shared_log_q = manager.Queue()
            self.logger.log(
                'info', f"Spawning {num_workers} worker processes...")
            results = list(tqdm(pool.starmap(process_pdf_worker, [
                           (fp, shared_log_q) for fp in filepaths]), total=len(filepaths), desc="Chunking New PDFs"))

        try:
            with open(self.artifacts['chunks'], 'r') as f:
                existing_chunks = json.load(f)
        except:
            existing_chunks = []
        all_chunks = [
            chunk for chunk in existing_chunks if chunk['source_file'] not in files_to_process]
        all_chunks.extend(
            [chunk for _, file_chunks in results for chunk in file_chunks])
        with open(self.artifacts['chunks'], 'w') as f:
            json.dump(all_chunks, f)

        try:
            with open(self.artifacts['cache_index'], 'r') as f:
                cache_index = json.load(f)
        except:
            cache_index = {}
        for f in files_to_process:
            cache_index[f] = self.get_file_hash(
                os.path.join(self.paths['library'], f))
        with open(self.artifacts['cache_index'], 'w') as f:
            json.dump(cache_index, f, indent=2)

    def stage_2_embed(self):
        self.logger.log(
            'info', "Stage 2 [Embedding]: Encoding text into vectors.")
        with open(self.artifacts['chunks'], 'r') as f:
            chunks = json.load(f)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        batch_size = 32
        if device == 'cuda':
            vram_gb = torch.cuda.get_device_properties(
                0).total_memory / (1024**3)
            batch_size = get_optimal_batch_size(vram_gb)
            self.logger.log(
                'info', f"Detected {vram_gb:.2f}GB VRAM. Using optimal batch size: {batch_size}")
        embeddings = model.encode(
            [c['text'] for c in chunks], batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
        np.save(self.artifacts['embeddings'], embeddings)
        self.logger.log('success', "Stage 2 Complete.")

    def stage_3_index(self):
        self.logger.log(
            'info', "Stage 3 [Indexing]: Building final FAISS index and metadata.")
        embeddings = np.load(self.artifacts['embeddings'])
        with open(self.artifacts['chunks'], 'r') as f:
            all_chunks = json.load(f)
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        faiss.write_index(index, self.artifacts['final_index'])
        with open(self.artifacts['final_meta'], 'w') as f:
            json.dump(all_chunks, f, indent=2)
        self.logger.log('success', "Stage 3 Complete.")


def main_process():
    root = tk.Tk()
    root.withdraw()
    log_queue = queue.Queue()
    viewer = LogViewer(root, log_queue)
    parser = argparse.ArgumentParser(description="Phoenix Build Pipeline")
    parser.add_argument('--force-rebuild', action='store_true')
    args = parser.parse_args()
    pipeline = PhoenixPipeline(log_queue, args)
    pipeline.run()
    # Signal the log viewer we are done, allowing it to prompt user.
    log_queue.put(
        "[SUCCESS] --- Pipeline has finished all tasks. You may close this window. ---")
    # Keep a reference to viewer and start the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    init(autoreset=True)  # Init colorama for main terminal
    print(f"{Fore.GREEN}{Style.BRIGHT}Starting Phoenix Pipeline... Detailed logs will appear in a separate window.{Style.RESET_ALL}")
    main_process()
