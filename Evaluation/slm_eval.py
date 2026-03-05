"""
SLM Evaluation
Evaluates the fine-tuned multi-task FinBERT model:
    1. Classification - accuracy, macro-F1, top-3 accuracy, confusion matrix
    2. Sentiment - MAE, Spearman correlation, improvement over base FinBERT
    3. Relevance - precision, recall, F1, AUC-ROC
    4. Teacher agreement - how often SLM matches Claude's labels
    5. Latency - speed comparison vs API fallback
"""

import logging 
import json 
import time 
import asyncio 
from pathlib import Path 
from typing import Dict, List, Optional, Tuple, cast 
from dataclasses import dataclass, field 
from collections import Counter 

import numpy as np 

logger = logging.getLogger(__name__)

@dataclass 
class ClassificationMetrics:
    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    top_3_accuracy: float = 0.0
    per_class_f1: Dict[str, float] = field(default_factory = dict)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory = dict)
    most_confused_pairs: List[Tuple[str, str, int]] = field(default_factory = list)
    n_examples: int = 0

@dataclass
class SentimentMetrics:
    mae: float = 0.0
    rmse: float = 0.0
    spearman_correlation: float = 0.0
    pearson_correlation: float = 0.0
    directional_accuracy: float = 0.0 #agrees on positive vs negative
    improvement_over_baseline: float = 0.0 #vs the raw FinBERT model
    n_examples: int = 0 

@dataclass
class RelevanceMetrics:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc_roc: float = 0.0
    avg_precision: float = 0.0 #Area under precision-recall curve
    threshold: float = 0.5
    optimal_threshold: float = 0.5
    n_examples: int = 0
    n_positive: int = 0
    n_negative: int = 0

@dataclass
class TeacherAgreementMetrics:
    classification_agreement: float = 0.0
    classification_top3_agreement: float = 0.0
    sentiment_rank_agreement: float = 0.0 #spearman between SLM and Claude
    relevance_agreement: float = 0.0
    n_compared: int = 0

@dataclass
class LatencyMetrics:
    slm_single_ms: float = 0.0
    slm_batch_100_ms: float = 0.0
    slm_batch_500_ms: float = 0.0
    api_single_ms: float = 0.0
    api_estimated_100: float = 0.0
    speedup_factor: float = 0.0
    cost_per_1000_slm: float = 0.0 #$0 local
    cost_per_1000_api: float = 0.0 #estimated API cost 

@dataclass
class SLMEvalMetrics:
    """ Complete SLM evaluation results"""
    classification: ClassificationMetrics = field(default_factory = ClassificationMetrics)
    sentiment: SentimentMetrics = field(default_factory=SentimentMetrics)
    relevance: RelevanceMetrics = field(default_factory = RelevanceMetrics)
    teacher_agreement: TeacherAgreementMetrics = field(default_factory=TeacherAgreementMetrics)
    latency: LatencyMetrics = field(default_factory = LatencyMetrics)

class SLMEvaluator:
    """
    Evaluates the fine tuned SLM against held out test data
    Test data comes from the same Claude labled pipeline used for training
    """

    def __init__(self, test_data_dir: str = "training_data"):
        self.test_data_dir = Path(test_data_dir)

    def evaluate(
        self,
        model_path: str = "models/financial_slm",
        backbone: str = "ProsusAI/finbert",
    ) -> SLMEvalMetrics:
        """ Running full evaluation"""
        from SLM.model import SLMInference

        metrics = SLMEvalMetrics()

        #Load the model
        try: 
            slm = SLMInference(model_path = model_path, backbone_name=backbone)
        except Exception as e:
            logger.error(f"Cannot load SLM for evaluation: {e}")
            return metrics 
        
        #Loading the test data
        cls_examples = self._load_jsonl("classification_test.jsonl")
        sent_examples = self._load_jsonl("sentiment_test.jsonl")
        rel_examples = self._load_jsonl("relevance_test.jsonl")

        #If no test split exists, try using the last 10% of the training data
        if not cls_examples:
            cls_examples = self._load_jsonl("classification_train.jsonl", tail_pct = 0.1)
        if not sent_examples:
            sent_examples = self._load_jsonl("sentiment_train.jsonl", tail_pct = 0.1)
        if not rel_examples:
            rel_examples = self._load_jsonl("relevance_train.jsonl", tail_pct = 0.1)

        #Evaluate each task
        if cls_examples:
            metrics.classification = self._eval_classification(slm, cls_examples)
        if sent_examples:
            metrics.sentiment = self._eval_sentiment(slm, sent_examples)
        if rel_examples:
            metrics.relevance = self._eval_relevance(slm, rel_examples)

        #Teacher agreement (compare the SLM vs Claude on the same inputs)
        if cls_examples or sent_examples or rel_examples:
            metrics.teacher_agreement = self._eval_teacher_agreement(
                slm, cls_examples, sent_examples, rel_examples,
            )

        #latency benchmarks
        all_texts = (
            [e["text"] for e in cls_examples[:100]] + 
            [e["text"] for e in sent_examples[:100]] + 
            [e["text"] for e in rel_examples[:100]]
        )
        if all_texts:
            metrics.latency = self._eval_latency(slm, all_texts)

        return metrics 
    
    # -- Classification Evaluation --#

    def _eval_classification(
        self, slm, examples: List[Dict],
    ) -> ClassificationMetrics:
        """ Evaluating the classification head"""
        from SLM.model import CLASSIFICATION_LABELS

        metrics = ClassificationMetrics(n_examples=len(examples))
        y_true = []
        y_pred = []
        top3_correct = 0

        for ex in examples:
            true_label = ex["label"]
            text = ex["text"]

            output = slm.analyze(text)
            pred_label = output.predicted_category

            y_true.append(true_label)
            y_pred.append(pred_label)

            #Top 3 accuracy
            sorted_probs = sorted(
                output.category_probabilities.items(),
                key = lambda x: x[1], reverse = True,
            )
            top3_labels = [p[0] for p in sorted_probs[:3]]
            if true_label in top3_labels:
                top3_correct += 1

        #Accuracy 
        metrics.accuracy = round(
            sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true), 4
        )
        metrics.top_3_accuracy = round(top3_correct / len(y_true), 4)

        #Per class F1
        labels = list(set(y_true + y_pred))
        per_class = {}
        weighted_f1_sum = 0.0
        macro_f1_sum = 0.0

        for label in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_class[label] = round(f1, 4)
            support = sum(1 for t in y_true if t == label)
            weighted_f1_sum += f1 * support
            macro_f1_sum += f1 

        metrics.per_class_f1 = per_class
        metrics.macro_f1 = round(macro_f1_sum / max(len(labels), 1), 4)
        metrics.weighted_f1 = round(weighted_f1_sum / max(len(y_true), 1), 4)

        #Confusion matrix (condensed: only counts sells with count > 0)
        confusion = {}
        for true, pred in zip(y_true, y_pred):
            confusion.setdefault(true, {})
            confusion[true][pred] = confusion[true].get(pred, 0) + 1
        metrics.confusion_matrix = confusion 

        #Most confused pairs
        confused_pairs = []
        for true_label, preds in confusion.items():
            for pred_label, count in preds.items():
                if true_label != pred_label and count > 0:
                    confused_pairs.append((true_label, pred_label, count))
        confused_pairs.sort(key = lambda x: x[2], reverse = True)
        metrics.most_confused_pairs = confused_pairs[:10]

        logger.info(
            f"Classification: accuracy = {metrics.accuracy}, "
            f"macro F1 = {metrics.macro_f1}, top 3 = {metrics.top_3_accuracy}"
        )
        return metrics 
    
    # -- Sentiment Evaluation -- #

    def _eval_sentiment(self, slm, examples: List[Dict]) -> SentimentMetrics:
        """ Evaluating sentiment head"""
        metrics = SentimentMetrics(n_examples=len(examples))

        y_true = []
        y_pred = []

        for ex in examples:
            true_sent = ex["sentiment"]
            output = slm.analyze(ex["text"])
            y_true.append(true_sent)
            y_pred.append(output.sentiment_score)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        #MAE and RMSE
        metrics.mae = round(float(np.mean(np.abs(y_true - y_pred))), 4)
        metrics.rmse = round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4)

        #Directional accuracy (both agree on sign)
        same_sign = np.sign(y_true) == np.sign(y_pred)
        metrics.directional_accuracy = round(float(np.mean(same_sign)), 4)

        #Spearman rank correlation
        from scipy.stats import spearmanr, pearsonr
        spearman_result = spearmanr(y_true, y_pred)
        pearson_result = pearsonr(y_true, y_pred)
        metrics.spearman_correlation = round(self._statistic_to_float(spearman_result), 4)
        metrics.pearson_correlation = round(self._statistic_to_float(pearson_result), 4)

        logger.info(
            f"Sentiment: MAE = {metrics.mae}, Spearman = {metrics.spearman_correlation}, "
            f"directional = {metrics.directional_accuracy}"
        )
        return metrics 
    
    # -- Relevance Evaluation -- #
    def _eval_relevance(self, slm, examples: List[Dict]) -> RelevanceMetrics:
        """ Evaluating relevance head"""
        metrics = RelevanceMetrics(n_examples = len(examples))

        y_true = []
        y_scores = []

        for ex in examples:
            true_rel = 1 if ex.get("is_relevant", False) else 0
            output = slm.analyze(ex["text"])
            y_true.append(true_rel)
            y_scores.append(output.relevance_score)

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        metrics.n_positive = int(y_true.sum())
        metrics.n_negative = int((1 - y_true).sum())

        #At default threshold (0.5)
        y_pred = (y_scores >= 0.5).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics.precision = round(precision, 4)
        metrics.recall = round(recall, 4)
        metrics.f1 = round(f1, 4)

        #AUC ROC (manual trapezoidal)
        metrics.auc_roc = round(self._compute_auc(y_true, y_scores), 4)

        #Find optimal threshold (max F1)
        best_f1 = 0.0
        best_threshold = 0.5 
        for t in np.arange(0.1, 0.9, 0.05):
            preds = (y_scores >= t).astype(int)
            tp_t = np.sum((preds == 1) & (y_true == 1))
            fp_t = np.sum((preds == 1) & (y_true == 0))
            fn_t = np.sum((preds == 0) & (y_true == 1))
            p = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0 
            r = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
            f1_t = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1_t > best_f1:
                best_f1 = f1_t 
                best_threshold = t

        metrics.optimal_threshold = round(float(best_threshold), 2)

        logger.info(
            f"Relevance: P = {metrics.precision}, R = {metrics.recall}, "
            f"F1 = {metrics.f1}, AUC = {metrics.auc_roc}"
        )
        return metrics 
    
    @staticmethod
    def _compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """ Compute AUC ROC via trapezoidal rule"""
        sorted_indices = np.argsort(-y_scores)
        y_sorted = y_true[sorted_indices]

        n_pos = y_true.sum()
        n_neg = len(y_true) - n_pos 
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        tpr_list = []
        fpr_list = []
        tp = 0
        fp = 0 

        for label in y_sorted:
            if label == 1:
                tp += 1
            else: 
                fp += 1
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)

        #Trapezoidal rule 
        auc = 0.0
        for i in range(1, len(fpr_list)):
            auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
        return auc 
    
    # -- Teacher Agreement -- #
    def _eval_teacher_agreement(
        self, slm,
        cls_examples: List[Dict],
        sent_examples: List[Dict],
        rel_examples: List[Dict],
    ) -> TeacherAgreementMetrics:
        """
        Measure agreement between the SLM and Claude (the teacher).
        Claude's labels are the ground truth in the test data
        So teacher agreement = accuracy on the test set
        """
        metrics = TeacherAgreementMetrics() 

        #Classification: Exact match + top 3 match 
        if cls_examples:
            exact = 0
            top3 = 0 
            for ex in cls_examples:
                output = slm.analyze(ex["text"])
                if output.predicted_category == ex["label"]:
                    exact += 1
                sorted_probs = sorted(
                    output.category_probabilities.items(),
                    key = lambda x: x[1], reverse = True,
                )
                if ex["label"] in [p[0] for p in sorted_probs[:3]]:
                    top3 += 1
            metrics.classification_agreement = round(exact / len(cls_examples), 4)
            metrics.classification_top3_agreement = round(top3 / len(cls_examples), 4)

        #Sentiment: Rank correlation with teacher labels
        if sent_examples:
            from scipy.stats import spearmanr 
            teacher = [ex["sentiment"] for ex in sent_examples]
            student = [slm.analyze(ex["text"]).sentiment_score for ex in sent_examples]
            corr_result = spearmanr(teacher, student)
            metrics.sentiment_rank_agreement = round(
                self._statistic_to_float(corr_result), 4
            )

        #Relevance: Agreement on binary decision 
        if rel_examples:
            agree = 0
            for ex in rel_examples:
                output = slm.analyze(ex["text"])
                teacher_relevant = ex.get("is_relevant", False)
                if output.is_relevant == teacher_relevant:
                    agree += 1 
            metrics.relevance_agreement = round(agree / len(rel_examples), 4)

        metrics.n_compared = len(cls_examples) + len(sent_examples) + len(rel_examples)
        return metrics 
    
    # -- Latency Benchmarks -- #
    def _eval_latency(self, slm, texts: List[str]) -> LatencyMetrics:
        """ Benchmarks SLM inference speed"""
        metrics = LatencyMetrics()
        texts = texts[:500]

        #Single inference
        times_single = []
        for text in texts[:20]:
            start = time.perf_counter()
            slm.analyze(text)
            elapsed = (time.perf_counter() - start) * 1000
            times_single.append(elapsed)
        metrics.slm_single_ms = round(float(np.median(times_single)), 2)

        #Batch of 100
        if len(texts) >= 100:
            start = time.perf_counter()
            slm.analyze_batch(texts[:100])
            metrics.slm_batch_100_ms = round(
                (time.perf_counter() - start) * 1000, 2
            )

        #Batch of 500
        if len(texts) >= 500:
            start = time.perf_counter()
            slm.analyze_batch(texts[:500])
            metrics.slm_batch_500_ms = round(
                (time.perf_counter() - start) * 1000, 2
            )

        #Estimated API cost comparisoin
        #Haiku: ~0.25$/M input tokens, ~100 tokens per article
        metrics.cost_per_1000_slm = 0.0
        metrics.cost_per_1000_api = round(1000 * 100 * 0.25 / 1_000_000, 4)
        metrics.api_estimated_100 = round(100 * 500, 2) #100 calls * ~500ms each 
        metrics.speedup_factor = round(
            metrics.api_estimated_100 / max(metrics.slm_batch_100_ms, 1), 1
        )

        logger.info(
            f"Latency: single = {metrics.slm_single_ms} ms, "
            f"Batch100 = {metrics.slm_batch_100_ms} ms, "
            f"Speedup = {metrics.speedup_factor} x vs API"
        )
        return metrics 
    
    @staticmethod
    def _statistic_to_float(result: object) -> float:
        """Extract scalar statistic value from SciPy correlation outputs."""
        statistic = getattr(result, "statistic", None)
        if statistic is None:
            statistic = cast(Tuple[float, float], result)[0]
        statistic_arr = np.asarray(statistic, dtype = float)
        if statistic_arr.size == 0:
            return 0.0
        return float(statistic_arr.reshape(-1)[0])
    
    # -- Data Loading -- #
    def _load_jsonl(
            self, filename: str, tail_pct: float = 1.0,
    ) -> List[Dict]:
        """ Load JSONL file. If tail_pct < 1.0, return only the last N%"""
        filepath = self.test_data_dir / filename 
        if not filepath.exists():
            return []
        
        examples = []
        with open(filepath) as f:
            for line in f:
                examples.append(json.loads(line))

        if tail_pct < 1.0:
            start = int(len(examples) * (1 - tail_pct))
            examples = examples[start:]

        return examples
