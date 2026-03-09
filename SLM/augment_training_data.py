"""
Training Data Augmentation for Financial SLM
Generates synthetic article headlines for existing labeled moves.

Problem: 94% of training texts are just "TICKER moved X% on DATE" — no financial
language for the SLM to learn from. At inference, the model sees real article
headlines, creating a train/test distribution mismatch.

Solution: Use Claude to generate realistic article headlines that match each
label category. Each move gets 3-5 synthetic headlines, giving the SLM actual
financial text patterns to learn from.

This is NOT fabricating labels — the labels are already Claude-verified.
We're only generating the INPUT TEXT that matches those verified labels.
"""

import asyncio
import json
import logging
import argparse
import random
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timezone

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, SecretStr

from Data.settings import Settings
from SLM.model import CLASSIFICATION_LABELS, LABEL_TO_IDX, consolidate_label

logger = logging.getLogger(__name__)
settings = Settings()


class SyntheticHeadline(BaseModel):
    """A single synthetic headline with sentiment."""
    headline: str = Field(description="Realistic financial news headline, under 120 chars")
    sentiment: float = Field(description="Financial sentiment: -1 (very bearish) to 1 (very bullish)")


class SyntheticHeadlines(BaseModel):
    """Claude-generated synthetic headlines for a labeled move."""
    headlines: List[SyntheticHeadline] = Field(
        description="4 realistic financial news headlines with sentiment scores"
    )


async def generate_headlines_for_move(
    llm,
    ticker: str,
    company_name: str,
    label: str,
    pct_change: float,
    date: str,
    move_sigma: float,
) -> List[SyntheticHeadline]:
    """Generate synthetic article headlines for one labeled move."""
    generator = llm.with_structured_output(SyntheticHeadlines)

    direction = "surged" if pct_change > 0 else "plunged"
    abs_pct = abs(pct_change)

    prompt = f"""Generate 4 realistic financial news article headlines that would explain this stock move.
For each headline, also provide a financial sentiment score.

MOVE:
- Company: {company_name} ({ticker})
- Date: {date}
- Change: {pct_change:+.1f}% ({direction} {abs_pct:.1f}%)
- Category: {label}
- Magnitude: {move_sigma:.1f}σ ({"very unusual" if move_sigma > 3 else "notable"})

Requirements:
- Headlines should sound like real Bloomberg/Reuters/CNBC headlines
- Each headline should be a plausible cause or report of this specific move
- Include specific details (quarter numbers, dollar amounts, analyst names, metrics)
- Vary the style: some breaking-news, some analytical, some focused on specific data points
- Headlines should clearly reflect the '{label}' category
- Keep each headline under 120 characters
- Sentiment: -1 (very bearish) to 1 (very bullish) based on financial implications"""

    try:
        result = await generator.ainvoke([
            SystemMessage(content=(
                "You are a financial news headline generator. Create realistic, specific "
                "headlines that sound like they come from major financial news outlets. "
                "Never use generic or vague language. Include specific numbers, names, and metrics. "
                "Provide accurate sentiment scores reflecting the financial implication of each headline."
            )),
            HumanMessage(content=prompt),
        ])
        return result.headlines
    except Exception as e:
        logger.warning(f"Headline generation failed for {ticker}/{date}: {e}")
        return []


async def augment_training_data(
    input_dir: str = "training_data",
    output_dir: str = "training_data",
    headlines_per_move: int = 4,
    max_moves: int = 500,
) -> Optional[Dict[str, int]]:
    """
    Augment classification training data with synthetic headlines.
    Reads existing classification_train.jsonl, generates headlines,
    and writes augmented_classification_train.jsonl.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load existing classification examples
    cls_path = input_path / "classification_train.jsonl"
    if not cls_path.exists():
        logger.error(f"No classification data at {cls_path}")
        return

    existing = []
    with open(cls_path) as f:
        for line in f:
            existing.append(json.loads(line))

    logger.info(f"Loaded {len(existing)} existing classification examples")

    # Initialize Claude
    llm = ChatAnthropic(
        model_name=settings.FAST_MODEL,
        api_key=SecretStr(settings.ANTHROPIC_API_KEY),
        max_tokens_to_sample=1024,
        temperature=0.7,  # Higher temp for diverse headlines
        timeout=None,
        stop=None,
    )

    # Generate synthetic headlines for each move
    aug_cls = []
    aug_sent = []
    aug_rel = []
    moves_to_process = existing[:max_moves]

    # Collect all headlines for cross-move negative relevance sampling
    all_headlines_pool = []

    for i, ex in enumerate(moves_to_process):
        if (i + 1) % 25 == 0:
            logger.info(f"Progress: {i+1}/{len(moves_to_process)} moves augmented")

        # Consolidate the label
        label = consolidate_label(ex.get("label", "unknown"))
        label_idx = LABEL_TO_IDX.get(label)
        if label_idx is None:
            continue

        ticker = ex.get("ticker", "")
        # Extract company name from text if possible
        text = ex.get("text", "")
        company_name = ticker  # fallback
        if "(" in text:
            company_name = text.split("(")[0].strip()

        headline_objs = await generate_headlines_for_move(
            llm=llm,
            ticker=ticker,
            company_name=company_name,
            label=label,
            pct_change=ex.get("move_pct", 0),
            date=text.split(" on ")[-1].split(".")[0] if " on " in text else "recent",
            move_sigma=ex.get("move_sigma", 2.0),
        )

        if not headline_objs:
            continue

        for h in headline_objs[:headlines_per_move]:
            headline_text = h.headline
            sentiment_score = max(-1.0, min(1.0, h.sentiment))

            # ── Classification example (headline with move context) ──
            aug_text = (
                f"{company_name} ({ticker}) {ex.get('move_pct', 0):+.1f}% "
                f"({ex.get('move_sigma', 2.0):.1f}σ). "
                f"News: {headline_text}"
            )
            aug_cls.append({
                "text": aug_text,
                "label": label,
                "label_idx": label_idx,
                "confidence": ex.get("confidence", 0.5),
                "ticker": ticker,
                "move_pct": ex.get("move_pct", 0),
                "move_sigma": ex.get("move_sigma", 2.0),
                "is_synthetic": True,
            })

            # Also: headline-only classification example
            aug_cls.append({
                "text": headline_text,
                "label": label,
                "label_idx": label_idx,
                "confidence": ex.get("confidence", 0.5),
                "ticker": ticker,
                "move_pct": ex.get("move_pct", 0),
                "move_sigma": ex.get("move_sigma", 2.0),
                "is_synthetic": True,
            })

            # ── Sentiment example ──
            aug_sent.append({
                "text": headline_text,
                "sentiment": sentiment_score,
                "ticker": ticker,
                "is_synthetic": True,
            })

            # ── Relevance example (POSITIVE: this headline IS about this move) ──
            rel_text = (
                f"[MOVE: {ticker} {ex.get('move_pct', 0):+.1f}% "
                f"{ex.get('move_pct', 0) > 0 and 'up' or 'down'}] "
                f"{headline_text}"
            )
            aug_rel.append({
                "text": rel_text,
                "relevance": 0.85 + random.random() * 0.15,  # 0.85-1.0
                "is_relevant": True,
                "ticker": ticker,
                "move_pct": ex.get("move_pct", 0),
                "is_synthetic": True,
            })

            # Save for negative sampling pool
            all_headlines_pool.append({
                "headline": headline_text,
                "ticker": ticker,
                "move_pct": ex.get("move_pct", 0),
            })

        # Rate limit
        await asyncio.sleep(0.3)

    # ── Generate NEGATIVE relevance examples ──
    # Pair each move with headlines from DIFFERENT moves (not relevant)
    logger.info("Generating negative relevance examples...")
    neg_rel_count = 0
    for ex in moves_to_process:
        ticker = ex.get("ticker", "")
        move_pct = ex.get("move_pct", 0)
        direction = "up" if move_pct > 0 else "down"

        # Sample 2 headlines from OTHER tickers
        other_headlines = [h for h in all_headlines_pool if h["ticker"] != ticker]
        if len(other_headlines) >= 2:
            negatives = random.sample(other_headlines, min(2, len(other_headlines)))
            for neg in negatives:
                rel_text = (
                    f"[MOVE: {ticker} {move_pct:+.1f}% {direction}] "
                    f"{neg['headline']}"
                )
                aug_rel.append({
                    "text": rel_text,
                    "relevance": random.random() * 0.2,  # 0.0-0.2
                    "is_relevant": False,
                    "ticker": ticker,
                    "move_pct": move_pct,
                    "is_synthetic": True,
                })
                neg_rel_count += 1

    logger.info(f"Generated {len(aug_cls)} cls, {len(aug_sent)} sent, {len(aug_rel)} rel augmented examples")
    logger.info(f"Relevance: {len(aug_rel) - neg_rel_count} positive + {neg_rel_count} negative")

    # ── Combine original + augmented for each task ──

    # Classification: remap originals + add synthetic
    combined_cls = []
    for ex in existing:
        label = consolidate_label(ex.get("label", "unknown"))
        label_idx = LABEL_TO_IDX.get(label)
        if label_idx is not None:
            combined_cls.append({
                "text": ex["text"],
                "label": label,
                "label_idx": label_idx,
                "confidence": ex.get("confidence", 0.5),
                "ticker": ex.get("ticker", ""),
                "move_pct": ex.get("move_pct", 0),
                "move_sigma": ex.get("move_sigma", 2.0),
                "is_synthetic": False,
            })
    combined_cls.extend(aug_cls)
    random.shuffle(combined_cls)

    # Sentiment: load originals + add synthetic
    combined_sent = []
    orig_sent_path = input_path / "sentiment_train.jsonl"
    if orig_sent_path.exists():
        with open(orig_sent_path) as f:
            for line in f:
                ex = json.loads(line)
                ex["is_synthetic"] = False
                combined_sent.append(ex)
    combined_sent.extend(aug_sent)
    random.shuffle(combined_sent)

    # Relevance: load originals + add synthetic
    combined_rel = []
    orig_rel_path = input_path / "relevance_train.jsonl"
    if orig_rel_path.exists():
        with open(orig_rel_path) as f:
            for line in f:
                ex = json.loads(line)
                ex["is_synthetic"] = False
                combined_rel.append(ex)
    combined_rel.extend(aug_rel)
    random.shuffle(combined_rel)

    # ── Save all three files ──
    for name, data in [
        ("classification_train.jsonl", combined_cls),
        ("sentiment_train.jsonl", combined_sent),
        ("relevance_train.jsonl", combined_rel),
    ]:
        out_file = output_path / name
        with open(out_file, "w") as f:
            for ex in data:
                f.write(json.dumps(ex, default=str) + "\n")

    # ── Report ──
    from collections import Counter
    cls_labels = Counter(ex["label"] for ex in combined_cls)
    cls_synthetic = sum(1 for ex in combined_cls if ex.get("is_synthetic"))
    sent_synthetic = sum(1 for ex in combined_sent if ex.get("is_synthetic"))
    rel_synthetic = sum(1 for ex in combined_rel if ex.get("is_synthetic"))
    rel_positive = sum(1 for ex in combined_rel if ex.get("is_relevant"))

    logger.info(f"\n{'='*60}")
    logger.info(f"AUGMENTED DATASET SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Classification: {len(combined_cls)} total ({len(existing)} orig + {cls_synthetic} synthetic)")
    for label, count in cls_labels.most_common():
        logger.info(f"  {label}: {count}")
    logger.info(f"Sentiment:      {len(combined_sent)} total ({len(combined_sent)-sent_synthetic} orig + {sent_synthetic} synthetic)")
    logger.info(f"Relevance:      {len(combined_rel)} total ({len(combined_rel)-rel_synthetic} orig + {rel_synthetic} synthetic)")
    logger.info(f"  Relevant: {rel_positive}, Not relevant: {len(combined_rel)-rel_positive}")
    logger.info(f"{'='*60}")

    return {
        "classification": len(combined_cls),
        "sentiment": len(combined_sent),
        "relevance": len(combined_rel),
    }


async def main():
    parser = argparse.ArgumentParser(description="Augment SLM training data with synthetic headlines")
    parser.add_argument("--input-dir", default="training_data", help="Input data directory")
    parser.add_argument("--output-dir", default="training_data", help="Output directory (overwrites classification_train.jsonl)")
    parser.add_argument("--headlines-per-move", type=int, default=4, help="Headlines to generate per move")
    parser.add_argument("--max-moves", type=int, default=500, help="Max moves to augment")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    counts = await augment_training_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        headlines_per_move=args.headlines_per_move,
        max_moves=args.max_moves,
    )
    if counts is None:
        logger.error("Augmentation failed.")
        return
    logger.info(f"Done. Classification: {counts['classification']}, Sentiment: {counts['sentiment']}, Relevance: {counts['relevance']}")


if __name__ == "__main__":
    asyncio.run(main())
