#!/usr/bin/env python3
"""
Backfill text_embedding into video_captions MongoDB collection.

video_captions has BLIP-generated captions like "a car is parked in a parking lot"
but no embeddings for semantic search. This script adds 768-dim embeddings
using all-mpnet-base-v2 to match the NLP search pipeline.

Usage:
    python backfill_video_caption_embeddings.py
    python backfill_video_caption_embeddings.py --dry-run
"""

import argparse
import sys
import numpy as np
from pymongo import MongoClient

MONGO_URI = "mongodb+srv://detectifai_user:DetectifAI123@cluster0.6f9uj.mongodb.net/detectifai"


def main():
    parser = argparse.ArgumentParser(description="Backfill embeddings into video_captions")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    args = parser.parse_args()

    print("Loading SentenceTransformer model (all-mpnet-base-v2)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")

    client = MongoClient(MONGO_URI)
    db = client.detectifai
    coll = db.video_captions

    # Find docs without embeddings
    query = {
        "$or": [
            {"text_embedding": {"$exists": False}},
            {"text_embedding": []},
            {"text_embedding": None},
        ]
    }
    docs = list(coll.find(query, {"_id": 1, "caption_id": 1, "sanitized_caption": 1, "raw_caption": 1, "video_id": 1, "frame_id": 1, "timestamp": 1}))
    total = len(docs)

    print(f"Found {total} video_captions without embeddings")

    if total == 0:
        print("Nothing to backfill!")
        client.close()
        return

    # Show samples
    print("\nSample captions to embed:")
    for d in docs[:5]:
        cap = d.get("sanitized_caption") or d.get("raw_caption", "")
        print(f"  [{d.get('caption_id', '?')[:12]}] {cap[:80]}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would backfill {total} documents. Exiting.")
        client.close()
        return

    # Batch process
    batch_size = 32
    updated = 0
    skipped = 0

    for i in range(0, total, batch_size):
        batch = docs[i:i + batch_size]
        texts = []
        for d in batch:
            text = d.get("sanitized_caption") or d.get("raw_caption", "")
            # Skip garbage captions
            if not text or text.strip().lower() in ("unable to generate caption", "error generating caption", ""):
                texts.append(None)
            else:
                texts.append(text)

        # Encode all non-None texts
        valid_texts = [t for t in texts if t is not None]
        if valid_texts:
            embeddings = model.encode(valid_texts, normalize_embeddings=True, batch_size=batch_size)
        else:
            embeddings = []

        emb_idx = 0
        for j, d in enumerate(batch):
            if texts[j] is None:
                skipped += 1
                continue

            emb = embeddings[emb_idx].tolist()
            emb_idx += 1

            coll.update_one(
                {"_id": d["_id"]},
                {"$set": {"text_embedding": emb}}
            )
            updated += 1

        print(f"  Processed {min(i + batch_size, total)}/{total} (updated: {updated}, skipped: {skipped})")

    print(f"\nDone! Updated: {updated}, Skipped (bad captions): {skipped}")

    # Verify
    with_emb = coll.count_documents({"text_embedding": {"$exists": True, "$ne": []}})
    print(f"video_captions with embeddings now: {with_emb}")

    client.close()


if __name__ == "__main__":
    main()
