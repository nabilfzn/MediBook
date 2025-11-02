# recommender.py
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

class BookRecommender:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.embeddings = None
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self._load_data()
        self._build_embeddings()

    def _load_data(self):
        self.df = pd.read_csv(self.csv_path)
        # Normalisasi kolom
        cols = [c.lower().strip() for c in self.df.columns]
        self.df.columns = cols
        if "title" not in self.df.columns:
            for c in cols:
                if "judul" in c or "title" in c:
                    self.df = self.df.rename(columns={c: "title"})
                    break
        if "available" not in self.df.columns:
            for c in cols:
                if "tersedia" in c or "avail" in c or "status" in c:
                    self.df = self.df.rename(columns={c: "available"})
                    break

        assert "title" in self.df.columns, "CSV harus punya kolom 'title' / 'judul'"
        assert "available" in self.df.columns, "CSV harus punya kolom 'available' / 'status/tersedia'"

        self.df["title"] = self.df["title"].fillna("").astype(str)
        self.df["available"] = self.df["available"].fillna("TIDAK").astype(str)
        self.df["is_available"] = self.df["available"].str.upper().isin(["YA", "AVAILABLE", "TRUE", "1"])

    def _embed_texts(self, texts):
        # Batch embedding dengan Gemini
        # genai.embed_content bisa dipanggil per item; untuk efisiensi, lakukan list-comprehension
        # (SDK saat ini tidak semua versi support batch besar dalam 1 call)
        vecs = []
        for t in texts:
            r = genai.embed_content(model=EMBED_MODEL, content=t)
            vecs.append(r["embedding"])
        return np.array(vecs, dtype=np.float32)

    def _build_embeddings(self):
        titles = self.df["title"].tolist()
        if len(titles) == 0:
            self.embeddings = np.zeros((0, 768), dtype=np.float32)  # dim tak kritikal di sini
            return
        self.embeddings = self._embed_texts(titles)

    def refresh(self):
        self._load_data()
        self._build_embeddings()

    def search(self, user_text: str, top_k=8, boost_availability=True):
        if self.embeddings is None or self.embeddings.shape[0] == 0:
            return []

        q_vec = self._embed_texts([user_text])
        sims = cosine_similarity(q_vec, self.embeddings)[0]

        scores = sims.copy()
        if boost_availability:
            boost = np.where(self.df["is_available"].values, 0.03, 0.0)
            scores = scores + boost

        idx = np.argsort(-scores)[:top_k]
        results = []
        for i in idx:
            results.append({
                "title": self.df.iloc[i]["title"],
                "available": "YA" if self.df.iloc[i]["is_available"] else "TIDAK",
                "score": float(scores[i]),
            })
        return results
