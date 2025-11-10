from flask import Flask, request, jsonify, render_template
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv

app = Flask(__name__)

# --- Load .env dan API Key Gemini ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # pastikan di .env, bukan langsung di code

# --- Baca data buku dan data UKS ---
df_book = pd.read_csv("data/Book22.csv")
df_book['mood'] = df_book['mood'].apply(lambda x: [m.strip().lower() for m in str(x).split(';')])

df_uks = pd.read_csv("data/uks.csv")
df_uks['gejala'] = df_uks['gejala'].apply(lambda x: [g.strip().lower() for g in str(x).split(';')])


# ====================== RENDER TEMPLATE =======================
@app.route("/chatbook")
def chatbook():
    return render_template("chatbook.html")

@app.route("/chatmedi")
def chatmedi():
    return render_template("chatmedi.html")

@app.route('/')
def home():
    return """
    <h2>Chat Buku & Chat Medis (Gemini)</h2>
    <ul>
      <li><a href='/chatbook'>Chat Buku</a></li>
      <li><a href='/chatmedi'>Chat UKS</a></li>
    </ul>
    """


# ====================== CHAT BUKU =======================
def rekomendasi_buku(mood):
    hasil = df_book[df_book['mood'].apply(lambda ms: mood.lower() in ms)]
    if hasil.empty:
        hasil = df_book[df_book['mood'].apply(lambda ms: any(mood.lower() in m for m in ms))]
    return hasil.head(3).to_dict(orient='records')


@app.route('/chat', methods=['POST'])
def chat_book():
    data = request.get_json()
    user_input = data.get('pesan', '').strip()
    if not user_input:
        return jsonify({'error': 'pesan tidak boleh kosong'}), 400

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")

        # --- Deteksi mood ---
        mood_prompt = f"""
        Kamu adalah analis emosi.
        Dari kalimat berikut, tentukan satu kata mood utama (misal: senang, sedih, refleksi, marah, bahagia, motivasi, fokus, dll).
        Kalimat: "{user_input}"
        Jawab hanya satu kata.
        """
        mood_response = model.generate_content(mood_prompt)
        mood = mood_response.text.strip().lower()

        # --- Ambil rekomendasi buku ---
        buku_list = rekomendasi_buku(mood)
        if not buku_list:
            return jsonify({
                'respon': f"Aku paham kok, tapi sepertinya belum ada buku yang cocok dengan suasana '{mood}' kamu."
            })

        buku_str = "\n".join([f"- {b['Title']} ({b['available']})" for b in buku_list])

        # --- Balasan empatik ---
        final_prompt = f"""
        Pesan pengguna: "{user_input}"
        Mood pengguna: {mood}
        Buku yang cocok:
        {buku_str}

        Buat jawaban empatik dan hangat seperti teman bicara, 
        lalu sarankan buku-buku itu dengan cara alami. Jangan terlalu formal.
        """
        final_response = model.generate_content(final_prompt)

        return jsonify({
            'mood_terdeteksi': mood,
            'respon': final_response.text
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ====================== CHAT MEDICAL / UKS =======================
def rekomendasi_obat(gejala):
    hasil = df_uks[df_uks['gejala'].apply(lambda gs: gejala.lower() in gs)]
    if hasil.empty:
        hasil = df_uks[df_uks['gejala'].apply(lambda gs: any(gejala.lower() in g for g in gs))]
    return hasil.head(3).to_dict(orient='records')


@app.route('/chatmedi_api', methods=['POST'])
def chat_medi():
    data = request.get_json()
    user_input = data.get('pesan', '').strip()
    if not user_input:
        return jsonify({'error': 'pesan tidak boleh kosong'}), 400

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")

        # --- Deteksi gejala utama dari curhatan pengguna ---
        gejala_prompt = f"""
        Kamu adalah asisten medis sekolah.
        Dari kalimat berikut, tentukan gejala utama yang dirasakan pengguna 
        (misalnya: pusing, demam, batuk, perut sakit, gatal, luka, stres, lemas, maag, dll).
        Kalimat: "{user_input}"
        Jawab hanya satu kata gejala tanpa penjelasan.
        """
        gejala_response = model.generate_content(gejala_prompt)
        gejala = gejala_response.text.strip().lower()

        # --- Rekomendasi obat berdasarkan gejala ---
        obat_list = rekomendasi_obat(gejala)

        if not obat_list:
            return jsonify({
                'respon': f"Aku paham kamu sedang merasa {gejala}, tapi sepertinya belum ada obat yang cocok di UKS untuk itu."
            })

        obat_str = "\n".join([f"- {o['obat']} ({o['ketersediaan']})" for o in obat_list])

        # --- Jawaban empatik dan rekomendasi ---
        final_prompt = f"""
        Pengguna berkata: "{user_input}"
        Gejala terdeteksi: {gejala}
        Obat yang cocok:
        {obat_str}

        Buat balasan yang empatik, menenangkan, dan membantu.
        Sertakan rekomendasi obat-obatan itu secara alami.
        Jangan formal seperti dokter, tapi seperti teman UKS yang peduli.
        """
        final_response = model.generate_content(final_prompt)

        return jsonify({
            'gejala_terdeteksi': gejala,
            'respon': final_response.text
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==============================================================
if __name__ == '__main__':
    app.run(debug=True, port=5001)
