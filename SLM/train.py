"""
Training Script for the Multi-Task Financial SLM
Fine-tunes the shared backbone + three task heads on the generated training data
Supports multi-task training with uncertainty-based loss weighting
"""

import json 
import logging 
import argparse 
from pathlib import Path 
from typing import List, Dict, Optional
from datetime import datetime 

import torch 
from torch.utils.data. import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer

from SLM.model import (
    FinancialMultiTaskSLM, MultiTaskLoss,
    CLASSIFICATION_LABELS, LABEL_TO_IDX, NUM_CLASSES,
)

logger = logging.getLogger(__name__)

# -- Datasets -- #

class MultiTaskFinancialDataset(Dataset):
    """
    Combined dataset for all three tasks
    Each example has text + labels for one or more tasks
    """

    def __init__(
        self,
        cls_path: Optional[str] = None,
        sent_path: Optional[str] = None,
        rel_path: Optional[str] = None,
        tokenizer_name: str = "ProsusAI/finbert",
        max_length: int = 512,
    ):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length 
        self.examples = []

        #Load classification examples
        if cls_path and Path(cls_path).exists():
            with open(cls_path) as f:
                for line in f:
                    ex = json.loads(line)
                    self.examples.append({
                        "test": ex["text"],
                        "task": "classifiation",
                        "cls_label": ex["label_idx"],
                        "sent_label": None,
                        "rel_label": None,
                    })

        #Load sentiment examples
        if sent_path and Path(sent_path).exists():
            with open(sent_path) as f:
                for line in f:
                    ex = json.loads(line)
                    self.examples.append({
                        "text": ex["text"],
                        "task": "sentiment",
                        "cls_label": None,
                        "sent_label": ex["sentiment"],
                        "rel_label": None,
                    })

        #Load relevance examples
        if rel_path and Path(rel_path).exists():
            with open(rel_path) as f:
                for line in f:
                    ex = json.loads(line)
                    self.examples.append({
                        "text": ex["text"],
                        "task": "relevance",
                        "cls_label": None,
                        "sent_label": None,
                        "rel_label": 1.0 if ex["is_relevant"] else 0.0,
                    })
        
        logger.info(f"Loaded {len(self.examples)} total training examples")

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex["text"],
            truncation = True,
            max_length = self.max_length,
            padding = "max_length",
            return_tensors = "pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "task": ex["task"],
        }

        #Only inlude lables for the relevant task
        if ex["cls_label"] is not None:
            item["cls_label"] = torch.tensor(ex["cls_label"], dtype = torch.long)
        if ex["sent_label"] is not None:
            item["sent_label"] = torch.tensor(ex["sent_label"], dtype = torch.float)
        if ex["rel_label"] is not None:
            item["rel_label"] = torch.tensor(ex["rel_label"], dtype = torch.float)

        return item 
    
def collate_fn(batch):
    """ Custom collate that handles optional labels"""
    result = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }

    #Collect labels only for examples that have them
    cls_labels = [b["cls_label"] for b in batch if "cls_label" in b]
    sent_labels = [b["sent_label"] for b in batch if "sent_label" in b]
    rel_labels = [b["rel_label"] for b in batch if "rel_label" in b]

    if cls_labels:
        result["cls_labels"] = torch.stack(cls_labels)
    if sent_labels:
        result["sent_labels"] = torch.stack(sent_labels)
    if rel_labels:
        result["rel_labels"] = torch.stack(rel_labels)

    return result 

# -- Trainer --#

class SLMTrainer:
    """
    Trianing loop for the SLM
    Handles mixed-task batches, evaluation, and checkpointing
    """

    def __init__(
        self,
        model: FinancialMultiTaskSLM,
        train_dataset: MultiTaskFinancialDataset,
        val_dataset: Optional[MultiTaskFinancialDataset] = None,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        epochs: int = 10,
        device: str = "auto",
        output_dir: str = "models/financial_slm",
    ):
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.loss_fn = MultiTaskLoss().to(self.device)
        self.epochs = epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents = True, exist_ok = True)

        self.train_loader = DataLoader(
            train_dataset, batch_size = batch_size, shuffle = True,
            collate_fn = collate_fn, num_workers= 0,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size = batch_size, shuffle = False,
            collate_fn = collate_fn, num_workers = 0,
        ) if val_dataset else None

        #Optimizer: Different LR for backbone vs heads 
        backbone_params = list(model.backbone.parameters())
        head_params = (
            list(model.shared_projection.parameters()) + 
            list(model.sentiment_head.parameters()) + 
            list(model.relevance_head.parameters())
        )
        loss_params = list(self.loss_fn.parameters())

        self.optimizer = AdamW([
            {"params": backbone_params, "lr": learning_rate * 0.1}, #Lower learning rate for the pretrained backbone
            {"params": head_params, "lr": learning_rate},
            {"params": loss_params, "lr": learning_rate * 0.5}, #Task weight learning
        ], weight_decay = weight_decay)

        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max = epochs * len(self.train_loader),
        )

        self.best_val_loss = float("inf")
        self.history: List[Dict] = []

    def train(self):
        """ The full training loop"""
        param_info = self.model.get_param_count()
        logger.info(f"Model parameters: {param_info['total']:,} total, {param_info['trainable']:,} trainable")
        logger.info(f"Training on {self.device} for {self.epochs} epochs")
        logger.info(f"Task weights (learnable): classification, sentiment, relevance")

        for epoch in range(self.epochs):
            #Train
            train_metrics = self._train_epoch(epoch)

            #Validate
            val_metrics = {}
            if self.val_loader:
                val_metrics = self._validate(epoch)

            #Log
            self.history.append({
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics
            })

            task_weights = train_metrics.get("task_weights", {})
            logger.info(
                f"Epoch {epoch + 1} / {self.epochs} | "
                f"Train loss: {train_metrics['total']:.4f} | "
                f"Val loss: {val_metrics.get('total', 'N/A')} | "
                f"Weights: cls = {task_weights.get('classification', 0):.2f} "
                f"sent = {task_weights.get('sentiment', 0):.2f} "
                f"rel = {task_weights.get('relevance', 0):.2f}"
            )

            #Checkpoint best model
            val_loss = val_metrics.get("total", train_metrics["total"])
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best")
                logger.info(f" New best model saved (loss: {val_loss:.4f})")

        #Save final model
        self._save_checkpoint("final")
        self._save_history()
        logger.info(f"Training complete. Models saved to {self.output_dir}")
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """ One training epoch"""
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        accumulated_metrics = {}

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            #Forward pass (all tasks)
            outputs = self.model(input_ids, attention_mask)

            #Compute multi-task loss
            cls_targets = batch.get("cls_labels", None)
            sent_targets = batch.get("sent_labels", None)
            rel_targets = batch.get("rel_labels", None)

            if cls_targets is not None:
                cls_targets = cls_targets.to(self.device)
            if sent_targets is not None:
                sent_targets = sent_targets.to(self.device)
            if rel_targets is not None:
                rel_targets = rel_targets.to(self.device)

            loss, metrics = self.loss_fn(
                outputs, cls_targets, sent_targets, rel_targets,
            )

            #Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)
            self.optimizer.step()
            self.scheduler.step() 

            total_loss += metrics["total"]
            batch_count += 1

            #Accumulate per task metrics
            for k, v in metrics.items():
                if k not in ("total", "task_weights"):
                    accumulated_metrics[k] = accumulated_metrics.get(k, 0) + v
        avg_metrics = {k: v / batch_count for k, v in accumulated_metrics.items()}
        avg_metrics["total"] = total_loss / batch_count 
        avg_metrics["task_weights"] = metrics.get("task_weights", {})

        return avg_metrics 
    
    @torch.no_grad() 
    def _validate(self, epoch: int) -> Dict[str, float]:
        """ Validation pass"""
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        correct_cls = 0
        total_cls = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            outputs = self.model(input_ids, attention_mask)

            cls_targets = batch.get("cls_labels", None)
            sent_targets = batch.get("sent_labels", None)
            rel_targets = batch.get("rel_labels", None)

            if cls_targets is not None:
                cls_targets = cls_targets.to(self.device)
                preds = outputs["classificaiton_logits"].argmax(dim = -1)
                correct_cls += (preds == cls_targets).sum().item()
                total_cls += cls_targets.size(0)
            if sent_targets is not None:
                sent_targets = sent_targets.to(self.device)
            if rel_targets is not None:
                rel_targets = rel_targets.to(self.device)

            loss, metrics = self.loss_fn(outputs, cls_targets, sent_targets, rel_targets)
            total_loss += metrics["total"]
            batch_count += 1

        result = {"total": round(total_loss / max(batch_count, 1), 4)}
        if total_cls > 0:
            result["classification_accuracy"] = round(correct_cls / total_cls, 4)
        return result 
    
    def _save_checkpoint(self, name: str):
        torch.save(self.model.state_dict(), self.output_dir / f"model_{name}.pt")
        #Also saves as the default model.pt for inference
        if name == "best":
            torch.save(self.model.state_dict(), self.output_dir / "model.pt")

    def _save_history(self):
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent = 2)

def main():
    parser = argparse.ArgumentParser(description="Train Financial Multi-Task SLM")
    parser.add_argument("--data-dir", default="training_data", help="Training data directory")
    parser.add_argument("--output-dir", default="models/financial_slm", help="Model output directory")
    parser.add_argument("--backbone", default="ProsusAI/finbert", help="Backbone model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--freeze-layers", type=int, default=2, help="Freeze N backbone layers")
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    data_dir = Path(args.data_dir)

    # Build dataset
    full_dataset = MultiTaskFinancialDataset(
        cls_path=str(data_dir / "classification_train.jsonl"),
        sent_path=str(data_dir / "sentiment_train.jsonl"),
        rel_path=str(data_dir / "relevance_train.jsonl"),
        tokenizer_name=args.backbone,
    )

    # Train/val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    logger.info(f"Train: {train_size} examples, Val: {val_size} examples")

    # Build model
    model = FinancialMultiTaskSLM(
        backbone_name=args.backbone,
        freeze_backbone_layers=args.freeze_layers,
    )

    # Train
    trainer = SLMTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        output_dir=args.output_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
