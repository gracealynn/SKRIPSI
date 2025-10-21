# app.py
import os
import logging
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

from flask import Flask, request, render_template, url_for, redirect
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from dotenv import load_dotenv
import markdown

from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
import base64
import io

import requests
import urllib.parse
import re

# Optional LLM client (Gemini)
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ---------------- Config ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "resnetrms.h5"   # ganti sesuai nama model
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
try:
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    LOCAL_SAVE_ENABLED = True
except Exception:
    LOCAL_SAVE_ENABLED = False


ALLOWED_EXT = {".jpg", ".jpeg", ".png"}
TARGET_SIZE = (224, 224)
OVERLAY_LLM_MAX = 512

LABEL_LIST_PATH = BASE_DIR / "label_list.npy"
if LABEL_LIST_PATH.exists():
    LABEL_LIST = list(np.load(str(LABEL_LIST_PATH), allow_pickle=True))
else:
    LABEL_LIST = ["Berminyak", "Dark Spots", "Jerawat", "Kemerahan", "Kering", "Kerutan"]

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Load .env & configure LLM ----------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
llm_client = None
if genai is not None and GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        llm_client = genai.GenerativeModel("gemini-2.5-flash")
        logger.info("Gemini LLM configured (gemini-2.5-flash).")
    except Exception as e:
        logger.warning(f"Failed to configure Gemini: {e}")
        llm_client = None
else:
    logger.info("Gemini not configured.")

# ---------------- Load Keras model ----------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model .h5 tidak ditemukan di: {MODEL_PATH}")
logger.info("Loading Keras model...")
model = load_model(str(MODEL_PATH))
logger.info("Model loaded.")

# ---------------- Helper functions ----------------
def allowed_file(filename: str):
    return Path(filename).suffix.lower() in ALLOWED_EXT

def is_face_image(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Lebih toleran terhadap wajah sebagian
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,    # lebih sensitif, jarak antar ukuran kecil
        minNeighbors=2,      # lebih longgar (default 5)
        minSize=(40, 40)     # wajah kecil pun bisa dideteksi
    )
    
    # Deteksi wajah samping (profil)
    profiles = profile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=2,
        minSize=(40, 40)
    )
    
    # Gabungkan hasil
    total_faces = len(faces) + len(profiles)
    return total_faces > 0

def find_last_conv_layer(keras_model):
    for layer in reversed(keras_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    for layer in reversed(keras_model.layers):
        if "conv" in layer.name.lower():
            return layer.name
    raise ValueError("Tidak menemukan layer convolution.")

def _ensure_tensor(x):
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("Object is empty list/tuple where tensor expected.")
        return x[0]
    return x

def make_gradcam_overlay(keras_model, img_rgb_uint8, x_preprocessed, pred_class, alpha=0.4, layer_name=None):
    if layer_name is None:
        layer_name = find_last_conv_layer(keras_model)

    try:
        layer_output = keras_model.get_layer(layer_name).output
    except Exception as e:
        raise ValueError(f"Layer {layer_name} tidak ditemukan: {e}")

    grad_model = tf.keras.models.Model(keras_model.inputs, [layer_output, keras_model.output])

    x = tf.convert_to_tensor(x_preprocessed, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        conv_outputs = _ensure_tensor(conv_outputs)
        predictions = _ensure_tensor(predictions)
        loss = predictions[:, pred_class]

    grads = tape.gradient(loss, conv_outputs)
    grads = _ensure_tensor(grads)

    if grads is None:
        raise RuntimeError("Gradien None — tidak dapat menghitung gradien.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs_arr = conv_outputs[0].numpy()
    pooled_grads_arr = pooled_grads.numpy()

    weighted = conv_outputs_arr * pooled_grads_arr[np.newaxis, np.newaxis, :]

    heatmap = np.mean(weighted, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    h, w = img_rgb_uint8.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    img_bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color_bgr, alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return heatmap_resized, overlay_rgb

def save_resized_for_llm(overlay_rgb_uint8, path_out: Path, max_size=OVERLAY_LLM_MAX):
    pil = Image.fromarray(overlay_rgb_uint8)
    pil.thumbnail((max_size, max_size))
    pil.save(str(path_out), format="JPEG", quality=85)

# ---------------- Prompt builders ----------------
def build_explanation_prompt(pred_label: str):
    return f"""
Kamu adalah sistem analisis kulit profesional.

Hasil prediksi model: {pred_label}.

Instruksi:
1. Berikan penjelasan langsung mengenai kondisi kulit {pred_label} secara singkat, padat, dan informatif.
2. Gunakan informasi dari visualisasi Grad-CAM (area berwarna merah/oranye) untuk menjelaskan bagian wajah yang menjadi fokus model.
3. Jelaskan secara logis mengapa area tersebut berkaitan dengan kondisi kulit {pred_label}.
4. Hindari sapaan atau kalimat pembuka seperti 'Halo', 'Tentu saja', dan sejenisnya.
5. Gunakan gaya bahasa profesional, objektif, dan mudah dipahami pengguna umum.
"""

def build_recommendation_prompt(pred_label: str):
    return f"""
Kamu adalah sistem rekomendasi perawatan kulit profesional.

Hasil prediksi model: {pred_label}.

Tugasmu:
1. Rekomendasikan **zat aktif skincare** yang sesuai untuk kondisi kulit {pred_label}.
2. Berikan contoh **produk skincare nyata** (brand indonesia dan global) yang mengandung zat aktif tersebut.
   - Sebutkan nama produk. **PENTING: WAJIB bungkus nama produk dengan tag [PRODUK] dan [/PRODUK]**.
   - kisaran harga produk dalam Rp
   - Sebutkan zat aktif utama
   - Jelaskan singkat kenapa produk itu cocok dengan kondisi kulit {pred_label}
3. Hindari kalimat pembuka seperti 'Halo', 'Tentu saja', atau ekspresi percakapan lainnya.
4. Gunakan bahasa formal dan to the point.
5. Ingatkan bahwa ini hanya saran umum berbasis AI, bukan diagnosis medis.

Format keluaran:
- Daftar poin untuk zat aktif
- Daftar poin untuk produk skincare
- Nama Produk: [PRODUK]Paula's Choice Clinical 0.3% Retinol + 2% Bakuchiol Treatment[/PRODUK]
- Zat Aktif Utama: Retinol 0.3%, Bakuchiol 2%
- Penutup berupa disclaimer singkat
"""

def build_input_validation_prompt(filename: str):
    return f"""
Kamu adalah validator gambar untuk aplikasi analisis kulit wajah manusia.

Informasi gambar:
- Nama file: {filename}

Tugasmu:
1. Tentukan apakah gambar ini **menampilkan wajah manusia**, baik penuh maupun sebagian (close-up atau dari samping).
2. Gambar yang menampilkan hewan, boneka, karakter kartun, patung, atau benda lain **tidak boleh diterima**.
3. Jika gambar valid (ada wajah manusia), jawab **hanya**: "VALID - wajah manusia terdeteksi."
4. Jika gambar tidak valid (bukan wajah manusia), jawab **hanya**: "TIDAK VALID - bukan wajah manusia."
5. Jangan tambahkan penjelasan lain. Jawaban harus singkat dan tegas.
"""

#search link produk
def get_real_product_link(product_name: str):
    """
    Cari link e-commerce resmi menggunakan Google Custom Search API.
    """
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    cx = os.getenv("GOOGLE_SEARCH_CX")
    if not api_key or not cx:
        return None

    query = urllib.parse.quote(
        f"{product_name} site:sociolla.com OR site:tokopedia.com OR site:shopee.co.id OR site:lazada.co.id OR site:watsons.co.id"
    )
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}"

    try:
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if "items" in data and len(data["items"]) > 0:
            return data["items"][0]["link"]  # ambil hasil teratas
    except Exception as e:
        print(f"[GoogleSearchAPI] Error: {e}")
    return None


# Connect to MongoDB Atlas
client = MongoClient(MONGO_URI)
db = client["skinalyze_db"]
collection = db["analysis_history"]

# ---------------- Flask app ----------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="Tidak menemukan file upload.")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="Nama file kosong.")
        if not allowed_file(file.filename):
            return render_template("index.html", error="File harus jpg/jpeg/png.")

        filename = secure_filename(file.filename)
        base, _ = os.path.splitext(filename)
        save_path = UPLOAD_FOLDER / filename
        file.save(str(save_path))

        # ------------------ VALIDASI GAMBAR ------------------
        opencv_valid = is_face_image(str(save_path))
        llm_valid = False
        llm_feedback = "LLM tidak dijalankan."

        # Selalu jalankan LLM validasi
        if llm_client is not None:
            try:
                prompt = build_input_validation_prompt(filename)
                img_for_llm = Image.open(str(save_path))
                resp = llm_client.generate_content([prompt, img_for_llm])
                llm_feedback = (getattr(resp, "text", None) or str(resp)).strip()
                print("LLM feedback:", llm_feedback)  # debug log

                resp_text = llm_feedback.lower().strip()
                if resp_text.startswith("valid") and "tidak" not in resp_text:
                    llm_valid = True
            except Exception as e:
                llm_feedback = f"LLM gagal validasi input: {e}"

        # Gabungkan hasil dengan operasi AND
        if not (opencv_valid and llm_valid):
            error_message = "❌ Gambar tidak valid. Harap unggah foto wajah manusia (boleh sebagian)."
            print(f"[VALIDATION FAIL] OpenCV={opencv_valid}, LLM={llm_feedback}")
            return render_template("index.html", error=error_message)

        # ------------------ LANJUTKAN PREDIKSI ------------------
        img_orig = Image.open(str(save_path)).convert("RGB")
        img_orig_rgb = np.array(img_orig)

        pil_for_model = img_orig.resize(TARGET_SIZE)
        img_array = keras_image.img_to_array(pil_for_model)
        x = np.expand_dims(img_array.copy(), axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        pred_class = int(np.argmax(preds, axis=1)[0])
        pred_label = LABEL_LIST[pred_class] if pred_class < len(LABEL_LIST) else f"Class-{pred_class}"
        confidence = float(np.max(preds))

        try:
            heatmap_resized, overlay_rgb = make_gradcam_overlay(model, img_orig_rgb, x, pred_class)
        except Exception as e:
            logger.exception("Grad-CAM gagal")
            return render_template("index.html", error=f"Grad-CAM error: {e}")

        # ------------------ SIMPAN HASIL ------------------
        orig_save = UPLOAD_FOLDER / f"{base}_orig.jpg"
        heat_save = UPLOAD_FOLDER / f"{base}_heat.jpg"
        overlay_save = UPLOAD_FOLDER / f"{base}_overlay.jpg"
        overlay_llm_save = UPLOAD_FOLDER / f"{base}_overlay_llm.jpg"

        # Buat heatmap RGB untuk disimpan / dikonversi ke base64
        heat_uint8 = np.uint8(255 * heatmap_resized)
        heat_col_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heat_col_rgb = cv2.cvtColor(heat_col_bgr, cv2.COLOR_BGR2RGB)

        # ✅ Simpan ke local hanya jika environment mengizinkan
        if LOCAL_SAVE_ENABLED:
            img_orig.save(str(orig_save))
            Image.fromarray(heat_col_rgb).save(str(heat_save))
            Image.fromarray(overlay_rgb).save(str(overlay_save))
            save_resized_for_llm(overlay_rgb, overlay_llm_save)
        else:
            logger.warning("⚠️ Penyimpanan lokal dinonaktifkan — hanya menyimpan ke MongoDB.")

        # ------------------ LLM EXPLANATION + RECOMMENDATION ------------------
        explanation_text = None
        recommendation_text = None
        if llm_client is not None:
            try:
                prompt1 = build_explanation_prompt(pred_label)
                img_for_llm = Image.open(str(overlay_llm_save))
                resp1 = llm_client.generate_content([prompt1, img_for_llm])
                explanation_text = getattr(resp1, "text", None) or str(resp1)

                prompt2 = build_recommendation_prompt(pred_label)
                resp2 = llm_client.generate_content(prompt2)
                recommendation_text = getattr(resp2, "text", None) or str(resp2)

                # 🔍 Cari link nyata dari Google Search API
                product_names = re.findall(r"\[PRODUK\](.*?)\[/PRODUK\]", recommendation_text)
                
                for name in product_names:
                    real_link = get_real_product_link(name)
                    
                    # Tag asli yang akan diganti, contoh: "[PRODUK]Nama Produk[/PRODUK]"
                    original_tagged_string = f"[PRODUK]{name}[/PRODUK]"
    
                    if real_link:
                        # Jika link ditemukan, ganti tag dengan Markdown link yang dicetak tebal
                        replacement = f"**[{name}]({real_link}){{: target='_blank' }}**"
                    else:
                        # Jika link tidak ditemukan, hapus tag dan hanya cetak tebal nama produknya
                        replacement = f"**{name}**"
        
                    recommendation_text = recommendation_text.replace(original_tagged_string, replacement)

                # Ubah ke format HTML agar hyperlink aktif di template
                recommendation_text = markdown.markdown(recommendation_text, extensions=["extra"])


                # Format agar rapi di HTML
                explanation_text = markdown.markdown(explanation_text, extensions=["extra"])
                recommendation_text = markdown.markdown(recommendation_text, extensions=["extra"])
            except Exception as e:
                explanation_text = "LLM gagal menjelaskan prediksi."
                recommendation_text = "LLM gagal memberi rekomendasi."

        # ================== Simpan ke MongoDB ==================
        try:
            # Konversi gambar ke Base64
            def img_to_base64(pil_image):
                buffer = io.BytesIO()
                pil_image.save(buffer, format="JPEG")
                return base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Simpan gambar sebagai Base64
            original_b64 = img_to_base64(img_orig)
            heatmap_b64 = img_to_base64(Image.fromarray(heat_col_rgb))
            overlay_b64 = img_to_base64(Image.fromarray(overlay_rgb))

            doc = {
                "user_name": request.form.get("nama", "Anonim"),
                "prediction": pred_label,
                "original_b64": original_b64,
                "heatmap_b64": heatmap_b64,
                "overlay_b64": overlay_b64,
                "explanation": explanation_text,
                "recommendation": recommendation_text,
                "created_at": datetime.utcnow()
            }
            collection.insert_one(doc)

            logger.info("Hasil analisis berhasil disimpan ke MongoDB.")
        except Exception as e:
            logger.error(f"Gagal menyimpan ke MongoDB: {e}")

        return render_template(
            "hasil.html",
            prediction=pred_label,
            original_b64=original_b64,
            heatmap_b64=heatmap_b64,
            overlay_b64=overlay_b64,
            explanation=explanation_text,
            recommendation=recommendation_text
        )

    return render_template("index.html")

@app.route("/history")
def history():
    query = request.args.get("q", "")

    try:
        if query:
            histories = list(collection.find({
                "$or": [
                    {"user_name": {"$regex": query, "$options": "i"}},
                    {"prediction": {"$regex": query, "$options": "i"}},
                    {"recommendation": {"$regex": query, "$options": "i"}}
                ]
            }).sort("created_at", -1))
        else:
            histories = list(collection.find().sort("created_at", -1))

        for h in histories:
            h["_id"] = str(h["_id"])  # ubah ObjectId ke string agar bisa dikirim ke template

    except Exception as e:
        histories = []
        logger.error(f"Gagal mengambil data history: {e}")

    return render_template("history.html", histories=histories)

@app.route("/history/<history_id>")
def history_detail(history_id):
    try:
        history_data = collection.find_one({"_id": ObjectId(history_id)})
        if not history_data:
            return render_template("history_detail.html", error="Data tidak ditemukan.")

        history_data["_id"] = str(history_data["_id"])
        return render_template("history_detail.html", data=history_data)
    except Exception as e:
        logger.error(f"Gagal mengambil detail history: {e}")
        return render_template("history_detail.html", error="Terjadi kesalahan saat memuat data.")
 
@app.route("/test-gemini")
def test_gemini():
    if llm_client is None:
        return "Gemini not configured."
    try:
        resp = llm_client.generate_content("Halo, ini tes koneksi ke Gemini.")
        return getattr(resp, "text", str(resp))
    except Exception as e:
        return f"Gemini error: {e}"
    
@app.route("/test-db")
def test_db():
    try:
        count = collection.count_documents({})
        return f"Terhubung ke MongoDB Atlas ✅ (total {count} dokumen)"
    except Exception as e:
        return f"Gagal konek MongoDB ❌: {e}"
    
@app.route("/test-search/<product>")
def test_search(product):
    link = get_real_product_link(product)
    return link or "Tidak ditemukan link relevan"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)