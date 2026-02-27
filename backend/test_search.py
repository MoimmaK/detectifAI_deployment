"""Test the updated NLP search that queries both collections."""
from nlp_search.query_retreival import retrieve_by_threshold
from pymongo import MongoClient

client = MongoClient("mongodb+srv://detectifai_user:DetectifAI123@cluster0.6f9uj.mongodb.net/detectifai")
db = client.detectifai

queries = ["car", "accident", "parking lot", "smoke", "person"]

for query in queries:
    print(f"\n=== Search: '{query}' (threshold=0.3) ===")
    results = retrieve_by_threshold(db, query, threshold=0.3)
    print(f"Found {len(results)} results")
    for r in results[:5]:
        sim = r["similarity"]
        caption = r["caption"][:60] if r.get("caption") else "N/A"
        src = r.get("source", "?")
        vid = r.get("video_id", "?")
        print(f"  [{sim:.2%}] {caption} | source={src} | vid={vid}")

client.close()
print("\nDone!")
