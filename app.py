# app.py
import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import google.generativeai as genai
from recomender import BookRecommender

load_dotenv()

# CSV_PATH = os.getenv("BOOKS_CSV_PATH", "data/data.csv")
CSV_PATH = "data/data.csv"
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__, template_folder="templates", static_folder="static")
reco = BookRecommender(CSV_PATH)
model = genai.GenerativeModel(CHAT_MODEL)

SYSTEM_PROMPT = (
    "Kamu adalah asisten perpustakaan. "
    "Tugasmu: beri rekomendasi buku secara empatik dan natural dalam Bahasa Indonesia. "
    "Kamu hanya menerima judul dan status ketersediaan; boleh gunakan pengetahuan umum tentang judul, "
    "tapi jangan mengarang detail spesifik yang tidak umum/keliru. "
    "Tampilkan 3–5 rekomendasi, utamakan yang tersedia. "
    "Format singkat: judul (status), lalu satu kalimat alasan relevansinya. "
    "Akhiri dengan satu pertanyaan tindak lanjut."
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_text = (data.get("message") or "").strip()
    if not user_text:
        return jsonify({"reply": "Ceritakan topik yang ingin kamu baca dan bagaimana perasaanmu sekarang 😊"})

    candidates = reco.search(user_text, top_k=8, boost_availability=True)
    books_block = "\n".join([f"- {c['title']} (tersedia: {c['available']})" for c in candidates])

    user_prompt = f"""
{SYSTEM_PROMPT}

Pengguna berkata: "{user_text}"

Daftar kandidat buku (judul + ketersediaan):
{books_block}

Instruksi:
1) Pilih 3–5 judul paling relevan dengan topik/perasaan pengguna.
2) Utamakan yang tersedia (YA). Jika ada yang TIDAK, boleh jadi alternatif cadangan di akhir.
3) Untuk tiap judul, tuliskan 1 kalimat alasan singkat & natural (hindari detail spesifik yang tidak umum).
4) Bahasa hangat & ringkas.
5) Tutup dengan 1 pertanyaan tindak lanjut.
"""

    try:
        resp = model.generate_content(user_prompt)
        reply = (resp.text or "").strip()
        if not reply:
            raise ValueError("Empty response")
    except Exception:
        available_first = sorted(candidates, key=lambda x: x["available"] != "YA")
        top = available_first[:4]
        lines = [f"Berikut beberapa pilihan yang cocok menurutku:"]
        for t in top:
            avail = "Tersedia" if t["available"] == "YA" else "Tidak tersedia"
            lines.append(f"- {t['title']} ({avail})")
        lines.append("Kamu lebih suka bacaan ringan atau yang mendalam?")
        reply = "\n".join(lines)

    return jsonify({"reply": reply, "candidates": candidates})

@app.route("/refresh", methods=["POST"])
def refresh():
    try:
        reco.refresh()
        return jsonify({"ok": True, "count": len(reco.df)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = 5001
    app.run(host="0.0.0.0", port=port, debug=True)
