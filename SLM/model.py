"""
Multi-Task Financial Small Language Model (SLM)
Shared transformer backbone with three task-specific heads:
    1. Root Cause Classification (25 categories)
    2. Financial Sentiment Scoring (regression, -1 to 1)
    3. Article Relevance/Causality (binary: causal vs coincidental)

Architecture: FinBERT backbone -> task specific heads 

Why I'm using FinBERT over a smaller DistilBERT:
    - Pre-trained on 1.8M financial news articles + SEC filings
    - Token embeddings already encode financial semantics ("downgrade" != "upgrade")
    - Sentiment head benefits from FinBERT's native financial sentiment training
    - Classification head needs less fine-tuning since its just financial event categories
    and this already aligns with the domain that FinBERT understands
    - ~100M params (BERT-base) vs 66M (DistilBERT): About 2x slower but it is a better
    representation for this project. Should be sub-seond on my CPU
"""

import torch 
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Tuple, 
from dataclasses import dataclass 
from enum import Enum

# -- Task Definition -- #

CLASSIFICATION_LABELS = [
    "fed_announcement", "rate_decision", "trade_war", "sanctions",
    "military_conflict", "election", "bill_signing", "commodity_disruption",
    "regulatory_change", "earnings_surprise", "merger_acquisition",
    "currency_crisis", "sovereign_debt", "central_bank_policy",
    "pandemic_health", "tech_disruption", "climate_event", "labor_market",
    "infrastructure", "geopolitical_tension", "economic_data",
    "sector_rotation", "analyst_rating", "insider_activity", "unknown",
]

NUM_CLASSES = len(CLASSIFICATION_LABELS)
LABEL_TO_IDX = {label: i for i, label in enumerate(CLASSIFICATION_LABELS)}
IDX_TO_LABEL = {i: label for i, label in enumerate(CLASSIFICATION_LABELS)}

@dataclass
class SLMOutput:
    """ Output from the multi-task small language model"""
    #classification
    predicted_category: str 
    category_probabilities: Dict[str, float]
    classification_confidence: float 

    #Sentiment
    sentiment_score: float #-1 (bearish) to 1 (bullish)

    #Relevance
    relevance_score: float #0 to 1
    is_relevant: bool #relevance > threshold 


#-- Multi-Task Model -- #

class FinancialMultiTaskSLM(nn.Module):
    """
    Multi-task transformer for financial event understanding

    Shared backbone first encodes text, then the three heads produce:
        - Classification: softmax over 25 root cause categories
        - Sentiment: single scalar in [-1, 1] via tanh 
        - Relevance: binary probability via sigmoid 

    The shared representation learns the general financial language understanding, 
    while then each head specializes for its task. This is a more parameter-efficient
    method rather than using three separate models and also allows knowledge
    transfer between tasks
    """

    def __init__(
        self,
        backbone_name: str = "ProsusAI/finbert",
        hidden_dim: int = 768,
        head_hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone_layers: int = 0,
    ):
        
        super().__init__()

        # -- Shared Backbone -- #
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.backbone_name = backbone_name 

        #Optionally freeze the early backbone layers for faster fine-tuning
        if freeze_backbone_layers > 0:
            #FinBERT/BERT uses encoder.layer; DistilBERT uses transformer.layer
            layers = getattr(self.backbone, 'encoder', getattr(self.backbone, 'transformer', None))
            if layers and hasattr(layers, 'layer'):
                for i, layer in enumerate(layers.layer):
                    if i < freeze_backbone_layers:
                        for param in layer.parameters():
                            param.requires_grad = False 

        #-- Shared projection (backbone output -> shared representation) -- #
        self.shared_projection = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.LayerNorm(head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        #-- Classificaiton Head (25 CATEGORIES) -- #
        self.classification_head = nn.Sequential(
            nn.Linear(head_hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, NUM_CLASSES),
        )

        #-- Sentiment Head (regression: -1 to 1) -- #
        self.sentiment_head = nn.Sequential(
            nn.Linear(head_hidden_dim, head_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim // 2, 1),
            nn.Tanh(),
        )

        #-- Relevance Head (binary classification) --#
        self.relevance_head = nn.Sequential(
            nn.Linear(head_hidden_dim, head_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """ 
        Forward pass. If the task is specified, only compute that head.
        Otherwise compute all three 

        Params:
            - input_ids:
                - (batch, seq_len) token IDs
            - attention_mask:
                - (batch, seq_len) attention mask 
            - task:
                - "classification", "sentiment", "relevance", or None (all)

        Returns:
            - Dict with keys: 
                - Classification_logits, sentiment, relevance
        """

        #Backbone encoding
        backbone_output = self.backbone(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )

        #Use [CLS] token representation 
        cls_repr = backbone_output.last_hidden_state[:, 0, :]

        #Shared projection
        shared = self.shared_projection(cls_repr)

        outputs = {}

        if task is None or task == "classification":
            outputs["classification_logits"] = self.classification_head(shared)

        if task is None or task == "sentiment": 
            outputs["sentiment"] = self.sentiment_head(shared).squeeze(-1)

        if task is None or task == "relevance":
            outputs["relevance"] = self.relevance_head(shared).squeeze(-1)

        return outputs 
    
    def get_param_count(self) -> Dict[str, int]:
        """ Return parameter counts by component"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        shared_params = sum(p.numel() for p in self.shared_projection.parameters())
        cls_params = sum(p.numel() for p in self.classification_head.parameters())
        sent_params = sum(p.numel() for p in self.sentiment_head.parameters())
        rel_params = sum(p.numel() for p in self.relevance_head.parameters())

        return {
            "backbone": backbone_params,
            "shared_projection": shared_params,
            "classification_head": cls_params,
            "sentiment_head": sent_params,
            "relevance_head": rel_params,
            "total": backbone_params + shared_params + cls_params + sent_params + rel_params,
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
    
#-- Multi Task Loss --#

class MultiTaskLoss(nn.Module):
    """
    Combined loss for all three tasks with learnable task weights
    Uses uncertainty based weighting (Kendall et al., 2018) so that the model 
    learns how much to weight each task automatically
    """

    def __init__(self):
        super().__init__():

        #Learnable log-variance parameters for uncertainty weighting
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_sent = nn.Parameter(torch.zeros(1))
        self.log_var_rel = nn.Parameter(torch.zeros(1))

        self.cls_loss = nn.CrossEntropyLoss()
        self.sent_loss = nn.MSELoss()
        self.rel_loss = nn.BCELoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        cls_targets: Optional[torch.Tensor] = None,
        sent_targets: Optional[torch.Tensor] = None,
        rel_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        """
        Compute weighted multi-task loss.
        Only computes loss for tasks in which targets are provided
        """

        total_loss = torch.tensor(0.0, device = next(iter(outputs.values())).device)
        loss_dict = {}

        if cls_targets is not None and "classification_logits" in outputs:
            cls_l = self.cls_loss(outputs["classification_logits"], cls_targets)
            precision_cls = torch.exp(-self.log_var_cls)
            total_loss += precision_cls * cls_l + self.log_var_cls 
            loss_dict["classification"] = cls_l.item() 

        if sent_targets is not None and "sentiment" in outputs:
            sent_l = self.sent_loss(outputs["sentiment"], sent_targets)
            precision_sent = torch.exp(-self.log_var_sent)
            total_loss += precision_sent * sent_l + self.log_var_sent
            loss_dict["sentiment"] = sent_l.item() 

        if rel_targets is not None and "relevance" in outputs:
            rel_l = self.rel_loss(outputs["relevance"], rel_targets)
            precision_rel = torch.exp(-self.log_var_rel)
            total_loss += precision_rel * rel_l + self.log_var_rel 
            loss_dict["relevance"] = rel_l.item()

        loss_dict["total"] = total_loss.item()
        loss_dict["task_weights"] = {
            "classification": torch.exp(-self.log_var_cls).item(),
            "sentiment": torch.exp(-self.log_var_sent).item(),
            "relevance": torch.exp(-self.log_var_rel).item(),
        }

        return total_loss, loss_dict 
    

#-- Inference Wrapper --#

class SLMInference:
    """ 
    Inference wrapper for the fine-tuned multi-task SLM.
    Handles tokenization, batching, and output formatting. 
    """

    def __init__(
        self,
        model_path: str = "models/financial_slm",
        backbone_name: str = "ProsusAI/finbert",
        device: str = "auto",
        relevance_threshold: float = 0.5,
    ):
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        self.model = FinancialMultiTaskSLM(backbone_name = backbone_name)
        self.relevance_threshold = relevance_threshold 

        #Loading fine-tuned weights if available
        try:
            state_dict = torch.load(
                f"{model_path}/model.pt",
                map_location = self.device,
                weights_only = True,
            )
            self.model.load_state_dict(state_dict)
            print(f" Loaded fine-tuned SLM from {model_path}")
        except FileNotFoundError:
            print(f" No fine tuned weights at {model_path}, using the base model")
        
        self.model.to(self.device)
        self.model.eval() 

    @torch.no_grad()
    def analyze(self, text: str) -> SLMOutput:
        """ Run all three tasks on a single text input"""
        inputs = self.tokenizer(
            text, return_tensors = "pt", truncation = True,
            max_length = 512, padding = True,
        ).to(self.device)

        outputs = self.model(
            input_ids = inputs["inputs_ids"],
            attention_mask = inputs["attention_mask"],
        )

        #Classification
        logits = outputs["classification_logits"][0]
        probs = torch.softmax(logits, dim = -1)
        top_idx = probs.argmax().item()
        confidence = probs[top_idx].item()

        category_probs = {
            CLASSIFICATION_LABELS[i]: probs[i].item()
            for i in range(NUM_CLASSES)
        }

        #Sentiment
        sentiment = outputs["sentiment"][0].item()

        #Relevance
        relevance = outputs["relevaance"][0].item() 

        return SLMOutput(
            predicted_category = IDX_TO_LABEL[top_idx],
            category_probabilities=category_probs,
            classification_confidence=confidence,
            sentiment_score=round(sentiment, 4),
            relevance_score = round(relevance, 4),
            is_relevant = relevance >= self.relevance_threshold,
        )
    
    @torch.no_grad()
    def analyze_batch(self, texts: List[str]) -> List[SLMOutput]:
        """ Run all three tasks on a batch of texts"""
        inputs = self.tokenizer(
            texts, return_tensors = "pt", truncation = True,
            max_length = 512, padding = True,
        ).to(self.device)

        outputs = self.model(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
        )

        results = []
        batch_size = len(texts)

        for i in range(batch_size):
            logits = outputs["classification_logits"][i]
            probs = torch.softmax(logits, dim = -1)
            top_idx = probs.argmax().item() 

            category_probs = {
                CLASSIFICATION_LABELS[j]: probs[j].item()
                for j in range(NUM_CLASSES)
            }

            results.append(SLMOutput(
                predicted_category = IDX_TO_LABEL[top_idx],
                category_probabilities= category_probs,
                classification_confidence= probs[top_idx].item(),
                sentiment_score = round(outputs["sentiment"][i].item(), 4),
                relevance_score=round(outputs["relevance"][i].item(), 4),
                is_relevant = outputs["relevance"][i].item() >= self.relevance_threshold,
            ))

        return results 
    
    @torch.no_grad()
    def classify(self, text: str) -> Tuple[str, float]:
        """ Classification only (the fastest path)"""
        inputs = self.tokenizer(
            text, return_tensors = "pt", truncation = True,
            max_length = 512, padding = True,
        ).to(self.device)

        outputs = self.model(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            task = "classification",
        )

        probs = torch.softmax(outputs["classification_logits"][0], dim = -1)
        top_idx = probs.argmax().item()
        return IDX_TO_LABEL[top_idx], probs[top_idx].item()
    
    @torch.no_grad()
    def score_sentiment(self, text: str) -> float:
        """Sentiment only"""
        inputs = self.tokenizer(
            text, return_tensors = "pt", truncation = True,
            max_length = 512, padding = True,
        ).to(self.device)
                  
        outputs = self.model(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            task = "sentiment",
        )
        return outputs["sentiment"][0].item() 
    
    @torch.no_grad()
    def score_relevance(self, text: str) -> float:
        """Relevance only"""
        inputs = self.tokenizer(
            text, return_tensors = "pt", truncation = True,
            max_length = 512, padding = True,
        ).to(self.device)

        outputs = self.model(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            task = "relevance",
        )
        return outputs["relevance"][0].item()