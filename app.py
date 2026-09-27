import os

import faiss
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "PolicyPulse_Cleaned.csv")
INDEX_PATH = os.path.join(BASE_DIR, "PolicyPulse.index")
MODEL_NAME = "all-MiniLM-L6-v2"

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")

print(f"[startup] Loading data from {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)
df = df.fillna("")

print(f"[startup] Loading FAISS index from {INDEX_PATH} ...")
index = faiss.read_index(INDEX_PATH)

print(f"[startup] Loading embedding model '{MODEL_NAME}' ...")
model = SentenceTransformer(MODEL_NAME)
print("[startup] Ready.")

_all_categories = set()
for cell in df["schemeCategory"]:
    for cat in str(cell).split(","):
        cat = cat.strip()
        if cat:
            _all_categories.add(cat)
ALL_CATEGORIES = sorted(_all_categories)
ALL_LEVELS = sorted({str(v).strip() for v in df["level"] if str(v).strip()})


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "total_schemes": int(index.ntotal)})


@app.route("/api/meta", methods=["GET"])
def meta():
    return jsonify({"levels": ALL_LEVELS, "categories": ALL_CATEGORIES})


@app.route("/api/search", methods=["POST"])
def search():
    body = request.get_json(force=True, silent=True) or {}
    query = (body.get("query") or "").strip()
    k = int(body.get("k", 8))
    level_filter = (body.get("level") or "").strip()
    category_filter = (body.get("category") or "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400
    k = max(1, min(k, 20))

    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    search_k = min(index.ntotal, k * 10) if (level_filter or category_filter) else k
    distances, indices = index.search(query_embedding, search_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        row = df.iloc[idx]

        if level_filter and row["level"] != level_filter:
            continue
        if category_filter and category_filter not in [
            c.strip() for c in str(row["schemeCategory"]).split(",")
        ]:
            continue

        results.append({
            "scheme_name": row["scheme_name"],
            "slug": row["slug"],
            "details": row["details"],
            "benefits": row["benefits"],
            "eligibility": row["eligibility"],
            "application": row["application"],
            "documents": row["documents"],
            "level": row["level"],
            "category": row["schemeCategory"],
            "tags": row["tags"],
            "score": round(float(1 / (1 + dist)), 4),
        })
        if len(results) >= k:
            break

    return jsonify({"query": query, "count": len(results), "results": results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)