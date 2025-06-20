# adapter.py
import os
import sys
import json
import logging
import time
import copy
import torch
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, T5ForConditionalGeneration
from datasets import load_dataset
from model import MedAI_AdaptiveBrain # Import our custom adaptive brain

# --- Logger for GUI Communication ---

def setup_adapter_logger():
    """Sets up a dedicated logger that writes structured messages for the GUI."""
    logger = logging.getLogger('Adapter')
    if not logger.handlers: # Prevent adding handlers multiple times
        logger.setLevel(logging.DEBUG)
        # Force flushing for real-time updates in the GUI's log monitor
        handler = logging.FileHandler('adapter.log', 'w', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    return logger

def status_reporter(logger, key, value):
    """Writes a machine-readable status update to the log."""
    logger.info(f"STATUS|{key.upper()}|{value}")

# --- Core Evaluation and Adaptation Logic ---

def evaluate_model(model: T5ForConditionalGeneration, tokenizer, dataloader, device):
    """
    Evaluates the base model's performance on a fixed validation dataset.
    This provides the score needed to calculate the reward.

    For a real system, this would use a metric like ROUGE, BLEU, or F1-score on a medical Q&A set.
    For this example, we'll use the model's loss as an inverse proxy for performance (lower loss = higher score).
    """
    model.eval()
    total_eval_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_eval_loss += outputs.loss.item()
            
    avg_loss = total_eval_loss / len(dataloader)
    # Return inverse loss as performance score (higher is better)
    return -avg_loss


def run_adaptation_cycle(model: MedAI_AdaptiveBrain, logger: logging.Logger, device: str):
    """The main SEAL-inspired loop to adapt the AI to new knowledge."""

    # --- 1. Load Resources ---
    unprocessed_knowledge_path = os.path.join('cache', 'unprocessed_knowledge.json')
    if not os.path.exists(unprocessed_knowledge_path):
        logger.info("LOG|INFO|No new knowledge file found. Nothing to adapt to.")
        return

    with open(unprocessed_knowledge_path, 'r', encoding='utf-8') as f:
        new_chunks = json.load(f)

    if not new_chunks:
        logger.info("LOG|INFO|Knowledge queue is empty. Adaptation cycle complete.")
        return

    logger.info(f"LOG|INFO|Found {len(new_chunks)} new knowledge chunks to adapt to.")

    # --- 2. Setup Datasets & Optimizers ---
    
    # Load a fixed, high-quality validation set. This is the bedrock of the reward system.
    # 'pubmed_qa' is a real medical Q&A dataset. We use the 'pqa_l' (long-form answers) version.
    # Note: Downloading this may take a moment on first run.
    logger.info("LOG|INFO|Loading validation dataset (pubmed_qa)...")
    try:
        validation_dataset = load_dataset("pubmed_qa", "pqa_l", split="train[:5%]")
    except Exception as e:
        logger.info(f"LOG|ERROR|Failed to load pubmed_qa dataset. Aborting. Error: {e}")
        return
        
    def preprocess_function(examples):
        # A simple preprocessing function for the QA dataset
        inputs = [f"question: {q} context: {c[0] if c else ''}" for q, c in zip(examples["question"], examples["context"])]
        targets = [a if a else "No answer" for a in examples["long_answer"]]
        model_inputs = model.tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = model.tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    processed_val_dataset = validation_dataset.map(preprocess_function, batched=True, remove_columns=validation_dataset.column_names)
    processed_val_dataset.set_format(type="torch")
    val_dataloader = torch.utils.data.DataLoader(processed_val_dataset, batch_size=8)
    logger.info("LOG|SUCCESS|Validation dataset ready.")
    
    # Setup two optimizers: one for the fine-tuning of the base model, one for the RL update of the editor.
    params = model.get_trainable_parameters()
    base_optimizer = AdamW(params["base_model"], lr=5e-5)
    generator_optimizer = AdamW(params["edit_generator"], lr=1e-5) # Lower LR for the policy model

    # --- 3. The Reinforcement Learning + Fine-Tuning Loop ---
    model.train() # Set model to training mode
    initial_score = evaluate_model(model.base_model, model.tokenizer, val_dataloader, device)
    status_reporter(logger, "INITIAL_SCORE", f"{initial_score:.4f}")
    logger.info(f"LOG|INFO|Initial validation score (inverse loss): {initial_score:.4f}")
    
    for i, chunk in enumerate(tqdm(new_chunks, desc="Adapting to Knowledge Chunks")):
        status_reporter(logger, "PROGRESS", (i + 1) / len(new_chunks))
        
        # --- STEP A: Generate a Self-Edit (The RL "Action") ---
        # The edit_generator proposes its own fine-tuning data from the raw text chunk.
        with torch.no_grad(): # We don't need gradients for this generation step
            generated_qa_pair = model.generate_self_edit(chunk['text'])
        logger.info(f"LOG|DEBUG|Generated Edit: {generated_qa_pair}")
        
        # --- STEP B: Apply Edit and Evaluate to get REWARD ---
        # Create a temporary copy of the base model to trial the edit on.
        # This is critical - we don't want to corrupt the main model with bad edits.
        temp_base_model = copy.deepcopy(model.base_model)
        temp_optimizer = AdamW(temp_base_model.parameters(), lr=5e-5)

        # Fine-tune the temporary model on the self-generated QA pair
        qa_inputs = model.tokenizer([generated_qa_pair], return_tensors='pt', padding=True, truncation=True).to(device)
        temp_base_model.train()
        for _ in range(3): # Train for a few steps
            outputs = temp_base_model(input_ids=qa_inputs.input_ids, labels=qa_inputs.input_ids)
            loss = outputs.loss
            loss.backward()
            temp_optimizer.step()
            temp_optimizer.zero_grad()

        # The reward is the change in performance on the fixed validation set.
        score_after_edit = evaluate_model(temp_base_model, model.tokenizer, val_dataloader, device)
        reward = score_after_edit - initial_score
        status_reporter(logger, "REWARD", f"{reward:+.4f}")

        # --- STEP C: Update the Edit Generator (The RL Policy Update) ---
        # To do a real policy gradient update, we need the log probability of the generated action.
        # The `generate` function in transformers can be configured to output scores.
        # This is a simplification to illustrate the concept.
        # In REINFORCE algorithm, Loss = -log_prob(action) * Reward
        
        # A placeholder for a complex calculation. The key is that the generator_loss
        # is proportional to the negative reward. High reward -> low loss. Low reward -> high loss.
        log_prob_placeholder = torch.tensor(-1.0, requires_grad=True) # Placeholder value
        generator_loss = -log_prob_placeholder * reward 
        
        # We only want to update the generator if the loss makes sense (to avoid instability)
        if isinstance(generator_loss, torch.Tensor):
            generator_optimizer.zero_grad()
            generator_loss.backward() # This will only affect edit_generator params if correctly wired.
            generator_optimizer.step()

        # --- STEP D: Commit Beneficial Changes to the REAL Base Model ---
        if reward > 0:
            logger.info(f"LOG|SUCCESS|Positive reward ({reward:+.4f}). Committing adaptation to main brain.")
            # Fine-tune the *real* base model on the QA pair that proved to be beneficial.
            base_inputs = model.tokenizer([generated_qa_pair], return_tensors='pt', padding=True, truncation=True).to(device)
            for _ in range(3):
                outputs = model.base_model(input_ids=base_inputs.input_ids, labels=base_inputs.input_ids)
                loss = outputs.loss
                loss.backward()
                base_optimizer.step()
                base_optimizer.zero_grad()
            
            # Update the baseline score to the new, improved score
            initial_score = score_after_edit 
        else:
            logger.info(f"LOG|WARNING|Negative/Neutral reward ({reward:+.4f}). Discarding this adaptation.")

    # --- 4. Cleanup ---
    logger.info("LOG|SUCCESS|Full adaptation cycle complete.")
    final_score = evaluate_model(model.base_model, model.tokenizer, val_dataloader, device)
    logger.info(f"LOG|SUCCESS|Final validation score: {final_score:.4f}. Initial score was: {-initial_score:.4f}.") # Original score was negative loss
    status_reporter(logger, "FINAL_SCORE", f"{final_score:.4f}")
    
    # Clear the knowledge queue file so we don't re-process
    with open(unprocessed_knowledge_path, 'w', encoding='utf-8') as f:
        json.dump([], f)
    logger.info("LOG|INFO|Unprocessed knowledge queue has been cleared.")

# --- Main Execution Block ---
if __name__ == "__main__":
    logger = setup_adapter_logger()
    logger.info("LOG|INFO|AI Adaptation script activated.")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"LOG|INFO|Using device: {device}")

    try:
        # Load the entire adaptive brain. This could be slow.
        status_reporter(logger, "SETUP", "Loading MedAI Brain...")
        med_ai_brain = MedAI_AdaptiveBrain(device=device)
        med_ai_brain.to(device)
        logger.info("LOG|SUCCESS|MedAI Brain loaded successfully.")
        
        # In a real system, you would load a checkpoint here:
        # if os.path.exists('checkpoints/med_ai_brain.pth'):
        #     med_ai_brain.load_state_dict(torch.load('checkpoints/med_ai_brain.pth'))
        #     logger.info("LOG|INFO|Loaded checkpoint successfully.")

        run_adaptation_cycle(med_ai_brain, logger, device)

        # In a real system, you would save the final adapted model:
        # os.makedirs('checkpoints', exist_ok=True)
        # torch.save(med_ai_brain.state_dict(), 'checkpoints/med_ai_brain.pth')
        # logger.info("LOG|SUCCESS|Saved updated brain state to checkpoint.")
        
        logger.info("LOG|SUCCESS|Adapter process finished.")
        
    except Exception as e:
        logger.error(f"LOG|ERROR|A fatal error occurred in the adapter: {e}", exc_info=True)