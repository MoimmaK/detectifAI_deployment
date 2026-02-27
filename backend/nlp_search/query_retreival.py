"""nlp_search/query_retreival.py

Command-line utility to run a natural-language query against stored
captions in MongoDB and return matching keyframes/captions above a similarity threshold.

Behavior:
 - Connects to MongoDB (MONGO_URI via env)
 - Loads the SentenceTransformer model to encode the query
 - Loads caption embeddings from the `event_descriptions` collection
   (documents should include `description_id`, `caption`, `text_embedding`,
    `event_id`, and `video_reference`)
 - Computes cosine similarity between query embedding and stored embeddings
 - Returns only matches with similarity >= 0.85 (85%) by default
 - Results include: caption, similarity_score (0..1), event_id (if present),
   video reference, and timestamps (from `events` collection if event exists)

Usage:
  python query_retreival.py --query "fire in building"
  python query_retreival.py -q "dog sitting" --threshold 0.80 --json

"""

import os
import argparse
import json
from dotenv import load_dotenv
from pymongo import MongoClient
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError("Please install sentence-transformers: pip install sentence-transformers")


load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/detectifai")


def connect_db():
    client = MongoClient(MONGO_URI)
    db = client.get_default_database()
    return db


def load_caption_embeddings(db):
    """Load captions and embeddings from both `event_description` and `video_captions`.

    Merges results from:
      - event_description: behavior-level captions (e.g., "Accident behavior detected")
      - video_captions: frame-level BLIP captions (e.g., "a car is parked in a parking lot")

    Returns:
      docs: list of dicts with keys: description_id, caption, event_id, video_reference, source
      emb_matrix: np.ndarray shape (N, D) of float32 (normalized)
    """
    docs = []
    embeddings = []

    # --- 1. Load from event_description (behavior-level) ---
    coll_ed = db.get_collection("event_description")
    cursor_ed = coll_ed.find({"text_embedding": {"$exists": True, "$ne": []}}, {
        "_id": 0,
        "description_id": 1,
        "caption": 1,
        "event_id": 1,
        "text_embedding": 1,
        "video_reference": 1
    })

    for doc in cursor_ed:
        emb = doc.get("text_embedding")
        if not emb:
            continue
        try:
            arr = np.asarray(emb, dtype="float32")
            norm = np.linalg.norm(arr)
            if norm == 0:
                continue
            arr = arr / norm
            embeddings.append(arr)
            docs.append({
                "description_id": doc.get("description_id"),
                "caption": doc.get("caption"),
                "event_id": doc.get("event_id"),
                "video_reference": doc.get("video_reference"),
                "source": "event_description"
            })
        except Exception:
            continue

    # --- 2. Load from video_captions (frame-level BLIP captions) ---
    coll_vc = db.get_collection("video_captions")
    cursor_vc = coll_vc.find({"text_embedding": {"$exists": True, "$ne": []}}, {
        "_id": 0,
        "caption_id": 1,
        "sanitized_caption": 1,
        "raw_caption": 1,
        "video_id": 1,
        "frame_id": 1,
        "timestamp": 1,
        "text_embedding": 1,
    })

    for doc in cursor_vc:
        emb = doc.get("text_embedding")
        if not emb:
            continue
        try:
            arr = np.asarray(emb, dtype="float32")
            norm = np.linalg.norm(arr)
            if norm == 0:
                continue
            arr = arr / norm
            embeddings.append(arr)
            caption_text = doc.get("sanitized_caption") or doc.get("raw_caption", "")
            docs.append({
                "description_id": doc.get("caption_id"),
                "caption": caption_text,
                "event_id": None,
                "video_id": doc.get("video_id"),
                "frame_id": doc.get("frame_id"),
                "timestamp": doc.get("timestamp"),
                "video_reference": None,
                "source": "video_captions"
            })
        except Exception:
            continue

    if embeddings:
        emb_matrix = np.stack(embeddings, axis=0).astype("float32")
    else:
        emb_matrix = np.zeros((0, 0), dtype="float32")

    return docs, emb_matrix


def compute_similarities(q_emb, emb_matrix):
    """Compute cosine similarities between q_emb (D,) and emb_matrix (N, D)."""
    if emb_matrix.size == 0:
        return np.array([])
    # ensure normalized
    q = q_emb / (np.linalg.norm(q_emb) + 1e-12)
    sims = np.dot(emb_matrix, q.astype("float32"))
    return sims


def retrieve_by_threshold(db, query_text, threshold=0.5):
    """Retrieve captions with similarity above threshold.
    
    Args:
        db: MongoDB database connection
        query_text: Query string
        threshold: Similarity threshold (0..1), default 0.85 (85%)
    
    Returns:
        List of results sorted by similarity (descending)
    """    
    model = SentenceTransformer("all-mpnet-base-v2")
    q_emb = model.encode(query_text, normalize_embeddings=True).astype("float32")

    docs, emb_matrix = load_caption_embeddings(db)

    if emb_matrix.size == 0:
        print("No caption embeddings found in database. Run upload_captions.py first.")
        return []

    sims = compute_similarities(q_emb, emb_matrix)

    # Filter by threshold and sort descending
    mask = sims >= threshold
    idxs = np.where(mask)[0]
    idxs = idxs[np.argsort(-sims[idxs])]  # Sort by similarity descending

    results = []
    events_coll = db.get_collection("events")
    keyframes_coll = db.get_collection("keyframes")

    for idx in idxs:
        score = float(sims[idx])
        doc = docs[idx]
        source = doc.get("source", "event_description")

        # Attempt to fetch timestamps from events collection
        start_ts = None
        end_ts = None
        video_id = doc.get("video_id")
        video_reference = doc.get("video_reference")

        if doc.get("event_id"):
            ev = events_coll.find_one({"event_id": doc.get("event_id")}, {"_id": 0, "start_timestamp_ms": 1, "end_timestamp_ms": 1, "video_id": 1})
            if ev:
                start_ts = ev.get("start_timestamp_ms")
                end_ts = ev.get("end_timestamp_ms")
                video_id = video_id or ev.get("video_id")

        # For video_captions source, try to find the keyframe image in MinIO
        if source == "video_captions" and not video_reference:
            frame_id = doc.get("frame_id")
            if frame_id:
                # Try to find keyframe record for this frame
                kf = keyframes_coll.find_one(
                    {"frame_id": frame_id},
                    {"_id": 0, "minio_bucket": 1, "minio_object_name": 1, "timestamp_ms": 1}
                ) if keyframes_coll is not None else None
                if kf and kf.get("minio_object_name"):
                    video_reference = {
                        "bucket": kf.get("minio_bucket", "keyframes"),
                        "object_name": kf.get("minio_object_name")
                    }
                    start_ts = start_ts or kf.get("timestamp_ms")

            # Use timestamp from the caption if still missing
            if not start_ts and doc.get("timestamp"):
                try:
                    start_ts = int(float(doc.get("timestamp")) * 1000) if doc.get("timestamp") else None
                except (ValueError, TypeError):
                    pass

        result = {
            "description_id": doc.get("description_id"),
            "caption": doc.get("caption"),
            "event_id": doc.get("event_id"),
            "video_reference": video_reference,
            "video_id": video_id,
            "frame_id": doc.get("frame_id"),
            "start_timestamp_ms": start_ts,
            "end_timestamp_ms": end_ts,
            "similarity": score,
            "source": source
        }
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Query NLP captions and retrieve matching keyframes/events from DB with similarity >= threshold")
    parser.add_argument("--query", "-q", required=True, help="Query text")
    parser.add_argument("--threshold", "-t", type=float, default=0.85, help="Similarity threshold (0..1), default 0.85 (85%)")
    parser.add_argument("--json", action="store_true", help="Print results as JSON")
    args = parser.parse_args()

    # Validate threshold
    if not (0.0 <= args.threshold <= 1.0):
        print("Error: threshold must be between 0.0 and 1.0")
        return

    db = connect_db()
    results = retrieve_by_threshold(db, args.query, threshold=args.threshold)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        if not results:
            print(f"Query: {args.query}\nNo matches found with similarity >= {args.threshold:.0%}")
        else:
            print(f"Query: {args.query}\nFound {len(results)} match(es) with similarity >= {args.threshold:.0%}:")
            for i, r in enumerate(results, 1):
                sim = r.get("similarity", 0.0)
                start = r.get("start_timestamp_ms")
                end = r.get("end_timestamp_ms")
                vidref = r.get("video_reference") or {}
                video_obj = vidref.get("object_name") if isinstance(vidref, dict) else None
                print(f"[{i}] Score: {sim:.4f} ({sim:.0%}) | Caption: {r.get('caption')} | Video Obj: {video_obj} | start_ms: {start} | end_ms: {end}")


if __name__ == "__main__":
    main()
