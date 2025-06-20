# utils.py
import torch
import os
import psutil
import platform
import logging
import sys
import datetime
from pkg_resources import working_set, DistributionNotFound
import faiss
import json
from sentence_transformers import SentenceTransformer

# ==============================================================================
# 1. PROFESSIONAL LOGGING SETUP
# ==============================================================================


def setup_logger(level=logging.INFO):
    """Sets up a logger that outputs to both a file and the console."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"training_log_{timestamp}.log")

    logger = logging.getLogger("EmptyBrain")
    logger.setLevel(level)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    info_formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    debug_formatter = logging.Formatter(
        '%(asctime)s - [DEBUG] - %(filename)s:%(lineno)d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler(sys.stdout)
    if level == logging.DEBUG:
        console_handler.setFormatter(debug_formatter)
    else:
        console_handler.setFormatter(info_formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(debug_formatter)
    logger.addHandler(file_handler)

    return logger

# ==============================================================================
# 2. ADVANCED SYSTEM MONITOR
# ==============================================================================


class SystemMonitor:
    def __init__(self, logger):
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.static_info = self._get_static_info()
        self.display_summary()

    def _human_readable_size(self, size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB")
        i = int(abs(size_bytes).log(1024))
        p = 1024 ** i
        s = round(size_bytes / p, 2)
        return f"{s} {size_name[i]}"

    def _get_static_info(self):
        info = {
            "OS": f"{platform.system()} {platform.release()}",
            "CPU": platform.processor(),
            "CPU Cores": f"{psutil.cpu_count(logical=True)} (Phys: {psutil.cpu_count(logical=False)})",
            "Python": platform.python_version(),
            "PyTorch": torch.__version__,
            "Target Device": self.device.upper()
        }
        if self.device == "cuda":
            info["GPU"] = torch.cuda.get_device_name(0)
            info["CUDA Version"] = torch.version.cuda
            info["Total VRAM"] = self._human_readable_size(
                torch.cuda.get_device_properties(0).total_memory)
        return info

    def get_dynamic_stats(self):
        stats = {
            "RAM Usage": f"{self._human_readable_size(psutil.virtual_memory().used)} / {self._human_readable_size(psutil.virtual_memory().total)} ({psutil.virtual_memory().percent}%)"}
        if self.device == "cuda":
            vram_used = torch.cuda.memory_allocated(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory
            vram_percent = (vram_used / vram_total *
                            100) if vram_total > 0 else 0
            stats["VRAM Usage"] = f"{self._human_readable_size(vram_used)} / {self._human_readable_size(vram_total)} ({vram_percent:.2f}%)"
        return stats

    def display_summary(self):
        self.logger.info("=" * 80)
        self.logger.info("SYSTEM MONITOR".center(80))
        for key, value in self.static_info.items():
            self.logger.info(f"{key:<15}: {value}")
        self.logger.info("-" * 80)

# ==============================================================================
# 3. PROJECT INTEGRITY CHECKER
# ==============================================================================


class ProjectIntegrity:
    def __init__(self, logger):
        self.logger = logger
        self.required_dirs = [
            'data/text_corpus', 'data/library', 'knowledge_base', 'brain_memory', 'logs']
        self.required_files = [
            'knowledge_base/faiss_index.bin', 'requirements.txt']

    def check_all(self):
        self.logger.info("--- Performing Project Integrity Check ---")
        all_ok = self._check_directories() and self._check_files() and self._check_dependencies()
        if all_ok:
            self.logger.info("✅ Project Integrity: OK")
        else:
            self.logger.critical(
                "❌ Project Integrity Check FAILED. Resolve issues above and restart.")
            sys.exit(1)
        self.logger.info("-" * 80)

    def _check_directories(self):
        for d in self.required_dirs:
            if not os.path.isdir(d):
                self.logger.warning(
                    f"Directory missing: '{d}'. Creating it now.")
                os.makedirs(d)
        return True

    def _check_files(self):
        all_found = True
        if not os.path.exists('knowledge_base/faiss_index.bin'):
            self.logger.error(
                "FATAL: Knowledge base not found. Run 'build_rag_index.py' first.")
            all_found = False
        return all_found

    def _check_dependencies(self):
        try:
            with open('requirements.txt', 'r') as f:
                required = {line.strip().split(
                    '==')[0] for line in f if line.strip() and not line.startswith('#')}
        except FileNotFoundError:
            self.logger.error("FATAL: `requirements.txt` not found.")
            return False

        installed = {pkg.key for pkg in working_set}
        missing = required - installed
        if missing:
            self.logger.error(
                f"FATAL: Missing Python packages: {', '.join(missing)}")
            self.logger.error(
                f"Please install by running: pip install {' '.join(missing)}")
            return False
        return True

# ==============================================================================
# 4. INTELLIGENT KNOWLEDGE BASE (CONTEXT-AWARE RAG)
# ==============================================================================


class IntelligentKnowledgeBase:
    def __init__(self, device='cpu', logger=None):
        self.logger = logger or logging.getLogger("EmptyBrain")
        self.logger.info("Loading Intelligent Knowledge Base...")
        try:
            self.index = faiss.read_index('knowledge_base/faiss_index.bin')
            with open('knowledge_base/documents_meta.json', 'r') as f:
                self.docs_meta = json.load(f)
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        except Exception as e:
            self.logger.critical(
                f"FATAL: Could not load Knowledge Base. Did you run build_rag_index.py? Error: {e}")
            sys.exit(1)

    def search(self, query_text, k=5, context="balanced"):
        query_embedding = self.model.encode(
            [query_text], convert_to_tensor=True)
        search_k = k * 10
        distances, indices = self.index.search(
            query_embedding.cpu().numpy(), search_k)
        candidates = [self.docs_meta[i] for i in indices[0]]

        def get_score(doc_tuple):
            i, doc = doc_tuple
            base_score = 1.0 / (1.0 + distances[0][i])
            boost = 1.0
            if context == "deep_dive":
                if doc['type'] == 'reference_tome':
                    boost = 1.5
                elif doc['type'] == 'student_textbook':
                    boost = 1.2
            elif context == "quick_fact":
                if doc['type'] == 'clinical_handbook':
                    boost = 2.0
                elif doc['type'] == 'clinical_textbook':
                    boost = 1.3
            return base_score * boost

        ranked_indices = sorted(enumerate(candidates),
                                key=get_score, reverse=True)
        return [c for i, c in ranked_indices[:k]]

# ==============================================================================
# 5. ROBUST BRAIN CHECKPOINTER
# ==============================================================================


class BrainCheckpointer:
    def __init__(self, model, optimizer, scheduler, logger, checkpoint_dir='brain_memory'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.checkpoint_dir = checkpoint_dir
        self.best_metric = float('inf')

    def save(self, epoch, current_metric):
        state = {'epoch': epoch, 'model_state': self.model.state_dict(), 'optimizer_state': self.optimizer.state_dict(),
                 'scheduler_state': self.scheduler.state_dict(), 'best_metric': self.best_metric}
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(state, latest_path)
        self.logger.info(f"Saved latest checkpoint to '{latest_path}'")
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pt')
            torch.save(state, best_path)
            self.logger.info(
                f"🏆 New best model! Saved to '{best_path}' (Loss: {current_metric:.4f})")

    def load(self, resume_from='latest_checkpoint.pt'):
        load_path = os.path.join(self.checkpoint_dir, resume_from)
        if not os.path.exists(load_path):
            self.logger.warning("Checkpoint not found. Starting from scratch.")
            return 0
        try:
            checkpoint = torch.load(load_path, map_location=self.model.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            if self.scheduler and 'scheduler_state' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.best_metric = checkpoint.get('best_metric', float('inf'))
            start_epoch = checkpoint['epoch'] + 1
            self.logger.info(
                f"✅ Resumed training from '{load_path}' at epoch {start_epoch}. (Best loss: {self.best_metric:.4f})")
            return start_epoch
        except Exception as e:
            self.logger.error(
                f"Failed to load checkpoint '{load_path}': {e}. Starting fresh.")
            return 0