# train.py
import torch
import os
import sys
import time
import argparse
import logging
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from itertools import chain
from PIL import Image

# Import local components
from model import EmptyBrainAI
from utils import IntelligentKnowledgeBase, BrainCheckpointer, SystemMonitor

# A dedicated file logger that writes structured messages


def setup_trainer_logger():
    # Force flushing
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    logger = logging.getLogger('Trainer')
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler('trainer.log', 'w')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def status_reporter(logger, key, value):
    """Writes a structured, machine-readable status update to the log."""
    logger.info(f"STATUS|{key.upper()}|{value}")

# Dummy dataset for completeness


class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self): self.questions, self.labels = ["Dummy Q"]*100, [0]*100
    def __len__(self): return 100

    def __getitem__(self, i): img = torch.randn(
        3, 224, 224); return (self.questions[i], img, self.labels[i])


if __name__ == "__main__":
    logger = setup_trainer_logger()
    logger.info("LOG|INFO|Trainer script activated.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Components initialization
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    knowledge_library = IntelligentKnowledgeBase(
        device=device, logger=None)  # The trainer logs its own stuff
    model = EmptyBrainAI(num_classes=10, device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_fn = torch.nn.CrossEntropyLoss()
    data_loader = torch.utils.data.DataLoader(
        MedicalDataset(), batch_size=4, shuffle=True)
    checkpointer = BrainCheckpointer(
        model, optimizer, scheduler, None)  # Pass None for logger
    start_epoch = checkpointer.load()

    logger.info("LOG|INFO|Beginning training loop...")
    for epoch in range(start_epoch, 50):
        status_reporter(logger, "EPOCH", f"{epoch+1} / 50")
        total_loss = 0

        for i, (questions, x_img, y_true) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}", file=sys.stdout)):
            status_reporter(logger, "PROGRESS", (i + 1) / len(data_loader))
            retrieved_docs_batch = [
                knowledge_library.search(q, k=5) for q in questions]
            with torch.no_grad():
                query_embeddings = text_encoder.encode(
                    questions, convert_to_tensor=True, device=device)
                knowledge_embeddings = torch.randn(
                    4, 5, 384).to(device)  # Placeholder
            x_img, y_true = x_img.to(device), y_true.to(device)
            output = model(query_embeddings, x_img, knowledge_embeddings)
            loss = loss_fn(output, y_true)
            status_reporter(logger, "LOSS_BATCH", f"{loss.item():.4f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss/len(data_loader) if data_loader else 0
        logger.info(
            f"LOG|INFO|Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        status_reporter(logger, "LOSS_AVG", f"{avg_loss:.4f}")
        scheduler.step(avg_loss)

    logger.info("LOG|SUCCESS|Training Complete.")
