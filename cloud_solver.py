import os
import sys
import time
import json
import base64
import hashlib
import re
import threading
from datetime import datetime
from bottle import Bottle, request, response, static_file
from waitress import serve
import concurrent.futures
import cv2
import numpy as np
import requests

ai_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Wymuszenie kodowania UTF-8 dla konsoli
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# =========================================================================
# KONFIGURACJA ŚRODOWISKA I MODELI
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_PHOTOS_DIR = os.path.join(BASE_DIR, "final_photos")
CAPTURED_DIR = os.path.join(BASE_DIR, "captured_photos")
WATCH_TEMPLATE_PATH = os.path.join(BASE_DIR, "watch_template.html")
SOLUTION_FILE = os.path.join(BASE_DIR, "latest_solution.json")
LIVE_FRAME_PATH = os.path.join(FINAL_PHOTOS_DIR, "latest_live_frame.jpg")

os.makedirs(FINAL_PHOTOS_DIR, exist_ok=True)
os.makedirs(CAPTURED_DIR, exist_ok=True)

MOCK_MODE = True
OCR_MODEL = "claude-haiku-5-5"
SOLVER_MODEL = "claude-opus-5-5"
ANTHROPIC_API_KEY = ""

def reload_env():
    global MOCK_MODE, OCR_MODEL, SOLVER_MODEL, ANTHROPIC_API_KEY
    # 1. Odczyt ze zmiennych środowiskowych kontenera / systemu (Render, Docker, Hugging Face)
    if "MOCK_MODE" in os.environ:
        MOCK_MODE = (os.environ["MOCK_MODE"].strip().lower() in ("true", "1", "yes"))
    if "OCR_MODEL" in os.environ:
        OCR_MODEL = os.environ["OCR_MODEL"].strip()
    if "SOLVER_MODEL" in os.environ:
        SOLVER_MODEL = os.environ["SOLVER_MODEL"].strip()
    if "ANTHROPIC_API_KEY" in os.environ:
        ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"].strip()

    # 2. Opcjonalne nadpisanie z pliku .env (przydatne przy testach lokalnych)
    ENV_FILE = os.path.join(BASE_DIR, ".env")
    if os.path.exists(ENV_FILE):
        try:
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k == "MOCK_MODE": MOCK_MODE = (v.lower() in ("true", "1", "yes"))
                        elif k == "OCR_MODEL": OCR_MODEL = v
                        elif k == "SOLVER_MODEL": SOLVER_MODEL = v
                        elif k == "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY = v
        except Exception as e:
            print(f"[ENV] Błąd wczytywania .env: {e}")

reload_env()

latest_solution = {
    "status": "ready",
    "html": "",
    "time": time.strftime("%H:%M:%S"),
    "hash": "init",
    "guidance": "",
    "mock_mode": MOCK_MODE
}

if os.path.exists(SOLUTION_FILE):
    try:
        with open(SOLUTION_FILE, "r", encoding="utf-8") as f:
            latest_solution = json.load(f)
    except Exception:
        pass

burst_accumulator = {}
app = Bottle()

# =========================================================================
# PROMPTY DLA ETAPU 1 (HAIKU 5.5) I ETAPU 2 (OPUS 5.5)
# =========================================================================
AUDITOR_PROMPT = """Jesteś bezwzględnym, laboratoryjnym audytorem jakości materiałów egzaminacyjnych i transkrypcji OCR.
Dostajesz 3 wycinki pionowe (góra, środek, dół) zdjęcia arkusza egzaminacyjnego.
Twoim zadaniem jest:
1. Zidentyfikować wszystkie zadania/pytania na arkuszu.
2. Zdecydować bezlitośnie, które zadania są W 100% CZYTELNE (wszystkie dane, indeksy, wykresy i treść są ostre i nieucięte).
3. Które zadania są UCIĘTE na krawędzi lub ROZMYTE (brak pewności co do liczb).
4. Jeśli jakieś zadanie jest nieczytelne/ucięte, sformułować krótką wskazówkę po polsku, jak użytkownik ma przesunąć aparat (np. 'Skieruj aparat 5 cm niżej', 'Popraw ostrość').

ZWRÓĆ WYŁĄCZNIE CZYSTY KOD JSON w formacie:
{
  "legible_tasks": [
    {
      "task_number": 1,
      "title": "Krótki tytuł",
      "content": "Pełna przepisana treść zadania z wszystkimi danymi liczbowymi i jednostkami"
    }
  ],
  "unclear_tasks": [
    {
      "task_number": 2,
      "reason": "Ucięty dolny wzór na marginesie"
    }
  ],
  "camera_instruction": "Skieruj aparat 4 cm w dół"
}"""

OPUS_SOLVER_PROMPT = """Jesteś profesorem nauk technicznych, matematyki i teorii obwodów.
Twoim celem jest rozwiązanie zweryfikowanych zadań egzaminacyjnych w formacie karty wzorcowych odpowiedzi.

ZASADY FORMATOWANIA (BEZWZGLĘDNE):
1. ZERO pogawędek, powitań, komentarzy w stylu 'Cześć' lub 'Oto rozwiązanie'.
2. Zwróć czysty kod HTML do wstrzyknięcia na czarnym tle (#000000).
3. Do wzorów matematycznych używaj WYŁĄCZNIE składni KaTeX:
   - Wzory blokowe: $$ ... $$
   - Wzory w tekście: $ ... $
   - ZAKAZ pisania '130 / 20' - ZAWSZE pisz eleganckie ułamki: \\frac{130}{20}
   - Wszelkie całki, granice, macierze, jednostki zapisuj formalnie: \\int_{0}^{t}, \\lim_{x \\to \\infty}, \\text{MPa}
4. Jeśli zadanie wymaga schematu lub wykresu (np. obwód RLC, belka, wykres rozciągania), ZAWSZE wygeneruj profesjonalny, estetyczny wektorowy schemat inline <svg ...> z ciemnym tłem, wyraźnymi liniami (#38bdf8, #10b981) i czytelnymi podpisami elementów.
5. Każde zadanie zamknij w tagu: <article class="full-task"> ... </article> z nagłówkiem, danymi, krokami obliczeniowymi, schematem i ramką wyniku końcowego.
"""

def run_stage_1_auditor(crop_paths):
    reload_env()
    if not ANTHROPIC_API_KEY:
        print("[AUDITOR] Brak ANTHROPIC_API_KEY!")
        return None
    blocks = []
    for p in crop_paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}
                })
    blocks.append({"type": "text", "text": "Przeanalizuj wycinki i zwróć czysty JSON z audytem zadań."})
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": OCR_MODEL,
        "max_tokens": 4096,
        "system": AUDITOR_PROMPT,
        "messages": [{"role": "user", "content": blocks}]
    }
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=25)
        if r.status_code == 200:
            data = r.json()
            text = "".join(x.get("text", "") for x in data.get("content", []))
            clean = re.sub(r"^```(?:json)?\s*", "", text.strip())
            clean = re.sub(r"\s*```$", "", clean)
            try:
                return json.loads(clean)
            except Exception:
                m = re.search(r'(\{.*\})', clean, re.DOTALL)
                if m:
                    return json.loads(m.group(1))
                raise
        else:
            print(f"[AUDITOR] Błąd API {r.status_code}: {r.text}")
    except Exception as e:
        print(f"[AUDITOR] Błąd: {e}")
    return None

def run_stage_2_opus(legible_tasks):
    reload_env()
    if not ANTHROPIC_API_KEY or not legible_tasks:
        return ""
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    prompt_tasks = json.dumps(legible_tasks, ensure_ascii=False, indent=2)
    payload = {
        "model": SOLVER_MODEL,
        "max_tokens": 8192,
        "system": OPUS_SOLVER_PROMPT,
        "messages": [
            {"role": "user", "content": f"Oto czytelne zadania z arkusza. Rozwiąż je krok po kroku z KaTeX i SVG:\n\n{prompt_tasks}"}
        ]
    }
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            data = r.json()
            text = "".join(x.get("text", "") for x in data.get("content", []))
            clean = re.sub(r"^```(?:html)?\s*", "", text.strip())
            clean = re.sub(r"\s*```$", "", clean)
            return clean
        else:
            print(f"[OPUS] Błąd API {r.status_code}: {r.text}")
    except Exception as e:
        print(f"[OPUS] Błąd: {e}")
    return ""

def build_guidance_banner(unclear_tasks, guidance):
    if not unclear_tasks and not guidance:
        return ""
    unclear_nums = ", ".join(str(u.get("task_number", "?")) for u in unclear_tasks)
    return f"""
    <div style="background: #451a03; border: 1.5px solid #f59e0b; border-radius: 8px; padding: 10px; margin-bottom: 12px; font-size: 11px;">
      <div style="color: #fbbf24; font-weight: 800; margin-bottom: 4px;">⚠️ UWAGA: CZĘŚĆ ZADAŃ WYMAGA POWTÓRZENIA ZDJĘCIA</div>
      <div style="color: #fde68a;">Zadania nieczytelne/ucięte: <strong>{unclear_nums}</strong></div>
      <div style="color: #fef3c7; margin-top: 4px;">Wskazówka: <strong>{guidance}</strong></div>
    </div>
    """

# =========================================================================
# MAKIETA ROZWIĄZANIA (MOCK ENGINE - ZERO TOKENÓW, PERFEKCYJNY KATEX I SVG)
# =========================================================================
def generate_mock_solution(crop_paths, capture_time):
    thumb_b64 = ""
    if len(crop_paths) > 1 and os.path.exists(crop_paths[1]):
        with open(crop_paths[1], "rb") as f:
            thumb_b64 = base64.b64encode(f.read()).decode("utf-8")

    img_html = f"<img src='data:image/jpeg;base64,{thumb_b64}' style='max-width: 100%; height: auto; border-radius: 4px; border: 1px solid #334155; display: block; margin: 0 auto;' />" if thumb_b64 else "<div style='color: #64748b;'>Brak zdjęcia</div>"

    html = """
    <div style="background: #022c22; border: 1.5px solid #10b981; border-radius: 8px; padding: 8px 10px; margin-bottom: 12px; font-size: 11px; display: flex; justify-content: space-between; align-items: center;">
      <span style="color: #6ee7b7; font-weight: 800;">⚡ MAKIETA TESTOWA ONLINE (KOSZT: 0,00 ZŁ)</span>
      <span style="color: #a7f3d0;">Odebrano: __CAPTURE_TIME__</span>
    </div>

    <div style="background: #090d16; border: 1px solid #1e293b; border-radius: 8px; padding: 6px; margin-bottom: 14px; text-align: center;">
      <div style="font-size: 10px; color: #94a3b8; font-weight: bold; margin-bottom: 4px; text-transform: uppercase;">Odebrany kadr z kamery XIAO (Tri-Crop)</div>
      __IMG_HTML__
    </div>

    <article class="full-task">
      <div class="task-top-bar">
        <span class="task-header-num">Zadanie 1</span>
        <span style="font-size: 10px; background: #0c2135; color: #38bdf8; padding: 2px 6px; border-radius: 4px; font-weight: bold;">STAN NIEUSTALONY RLC</span>
      </div>
      <div class="task-title">Załączenie źródła stałego na obwód z cewką i kondensatorem (Rachunek Operatorowy)</div>

      <div class="schematic-box">
        <div class="schematic-tag tag-po">Schemat operatorowy po komutacji (t &ge; 0)</div>
        <svg viewBox="0 0 540 180" xmlns="http://www.w3.org/2000/svg">
          <rect width="540" height="180" fill="#040810" rx="8"/>
          <line x1="60" y1="90" x2="130" y2="90" stroke="#38bdf8" stroke-width="2.5"/>
          <circle cx="95" cy="90" r="22" stroke="#38bdf8" stroke-width="2" fill="#082f49"/>
          <path d="M 95 76 L 95 104 M 85 83 L 105 83" stroke="#38bdf8" stroke-width="2"/>
          <text x="95" y="130" fill="#94a3b8" font-size="12" text-anchor="middle" font-weight="bold">E(s) = E/s</text>
          <line x1="130" y1="90" x2="180" y2="90" stroke="#38bdf8" stroke-width="2.5"/>
          <rect x="180" y="78" width="60" height="24" stroke="#f59e0b" stroke-width="2" fill="#451a03"/>
          <text x="210" y="94" fill="#fbbf24" font-size="11" text-anchor="middle" font-weight="bold">R</text>
          <line x1="240" y1="90" x2="280" y2="90" stroke="#38bdf8" stroke-width="2.5"/>
          <path d="M 280 90 Q 290 68 300 90 Q 310 68 320 90 Q 330 68 340 90" stroke="#10b981" stroke-width="2.5" fill="none"/>
          <text x="310" y="62" fill="#34d399" font-size="11" text-anchor="middle" font-weight="bold">sL</text>
          <line x1="340" y1="90" x2="380" y2="90" stroke="#38bdf8" stroke-width="2.5"/>
          <line x1="380" y1="74" x2="380" y2="106" stroke="#ec4899" stroke-width="3"/>
          <line x1="388" y1="74" x2="388" y2="106" stroke="#ec4899" stroke-width="3"/>
          <text x="384" y="62" fill="#f472b6" font-size="11" text-anchor="middle" font-weight="bold">1/(sC)</text>
          <line x1="388" y1="90" x2="480" y2="90" stroke="#38bdf8" stroke-width="2.5"/>
          <line x1="480" y1="90" x2="480" y2="150" stroke="#38bdf8" stroke-width="2.5"/>
          <line x1="480" y1="150" x2="60" y2="150" stroke="#38bdf8" stroke-width="2.5"/>
          <line x1="60" y1="150" x2="60" y2="90" stroke="#38bdf8" stroke-width="2.5"/>
          <line x1="250" y1="110" x2="290" y2="110" stroke="#38bdf8" stroke-width="1.8" marker-end="url(#arr)"/>
          <text x="270" y="126" fill="#38bdf8" font-size="11" text-anchor="middle">i(t)</text>
          <defs>
            <marker id="arr" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 1 L 8 5 L 0 9 z" fill="#38bdf8"/>
            </marker>
          </defs>
        </svg>
      </div>

      <div class="data-badge-row">
        <span class="data-chip">$$E = 100\\text{ V}$$</span>
        <span class="data-chip">$$R = 20\\ \\Omega$$</span>
        <span class="data-chip">$$L = 0{,}1\\text{ H}$$</span>
        <span class="data-chip">$$C = 50\\ \\mu\\text{F}$$</span>
      </div>

      <div class="calc-step">
        <div class="step-num">Krok 1 &bull; Równanie operatorowe impedancji pętli</div>
        $$Z(s) = R + sL + \\frac{1}{sC} = \\frac{s^2 L C + s R C + 1}{sC}$$
        Prąd operatorowy wynosi:
        $$I(s) = \\frac{E(s)}{Z(s)} = \\frac{E / s}{\\frac{s^2 L C + s R C + 1}{sC}} = \\frac{E / L}{s^2 + \\frac{R}{L}s + \\frac{1}{LC}}$$
      </div>

      <div class="calc-step">
        <div class="step-num">Krok 2 &bull; Wyznaczenie pierwiastków wielomianu charakterystycznego</div>
        Współczynnik tłumienia oraz pulsacja rezonansowa:
        $$\\alpha = \\frac{R}{2L} = \\frac{20}{2 \\cdot 0{,}1} = 100\\text{ s}^{-1}, \\quad \\omega_0^2 = \\frac{1}{LC} = \\frac{1}{0{,}1 \\cdot 50 \\cdot 10^{-6}} = 200\\ 000\\text{ rad}^2/\\text{s}^2$$
        Wyróżnik $\\Delta$:
        $$\\Delta = \\alpha^2 - \\omega_0^2 = 10\\ 000 - 200\\ 000 = -190\\ 000 < 0 \\implies \\text{stan oscylacyjny}$$
        Pulsacja drgań tłumionych $\\omega_d$:
        $$\\omega_d = \\sqrt{\\omega_0^2 - \\alpha^2} = \\sqrt{190\\ 000} \\approx 435{,}89\\text{ rad/s}$$
      </div>

      <div class="result-box">
        <div class="result-label">WYNIK KOŃCOWY (POSTAĆ ANALITYCZNA DLA t &ge; 0)</div>
        <div class="result-val">$$i(t) = 2{,}294 \\cdot e^{-100 t} \\cdot \\sin(435{,}89 t)\\text{ A}$$</div>
      </div>
    </article>
    """
    return html.replace("__CAPTURE_TIME__", capture_time).replace("__IMG_HTML__", img_html)

# =========================================================================
# OBRÓBKA OBRAZU
# =========================================================================
def select_sharpest_frame(frames_bytes):
    if len(frames_bytes) == 1:
        return frames_bytes[0]
    best_frame = frames_bytes[0]
    max_score = -1.0
    for b in frames_bytes:
        arr = np.frombuffer(b, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
            gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            k = max(1, int(mag.size * 0.05))
            score = float(np.mean(np.partition(mag.flatten(), -k)[-k:]))
            if score > max_score:
                max_score = score
                best_frame = b
    return best_frame

def enhance_and_crop(raw_jpeg_bytes):
    nparr = np.frombuffer(raw_jpeg_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, []

    rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Balans bieli Gray-World
    result = rotated.copy().astype(np.float32)
    b, g, r = cv2.split(result)
    m_b, m_g, m_r = np.mean(b), np.mean(g), np.mean(r)
    m_gray = (m_b + m_g + m_r) / 3.0
    if m_b > 0 and m_g > 0 and m_r > 0:
        b = np.clip(b * (m_gray / m_b), 0, 255)
        g = np.clip(g * (m_gray / m_g), 0, 255)
        r = np.clip(r * (m_gray / m_r), 0, 255)
        result = cv2.merge([b, g, r]).astype(np.uint8)
    else:
        result = rotated

    # CLAHE na kanale L + Unsharp Mask (ostre litery i wzory)
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    l_blur = cv2.GaussianBlur(l_clahe, (0, 0), 2.0)
    l_sharp = cv2.addWeighted(l_clahe, 1.35, l_blur, -0.35, 0)
    final_full = cv2.cvtColor(cv2.merge([l_sharp, a, b_ch]), cv2.COLOR_LAB2BGR)

    full_path = os.path.join(FINAL_PHOTOS_DIR, "ideal_capture.jpg")
    cv2.imwrite(full_path, final_full, [cv2.IMWRITE_JPEG_QUALITY, 93])

    h, w = final_full.shape[:2]
    slices = [
        ("section_1_top.jpg", 0.0, 0.38),
        ("section_2_middle.jpg", 0.32, 0.70),
        ("section_3_bottom.jpg", 0.64, 1.00)
    ]
    crop_paths = []
    for name, y1_pct, y2_pct in slices:
        y1, y2 = int(h * y1_pct), int(h * y2_pct)
        crop = final_full[y1:y2, 0:w]
        c_path = os.path.join(FINAL_PHOTOS_DIR, name)
        cv2.imwrite(c_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 93])
        crop_paths.append(c_path)

    return full_path, crop_paths

# =========================================================================
# ENDPOINTY SERWERA
# =========================================================================
@app.route("/")
@app.route("/watch")
def serve_watch_ui():
    if os.path.exists(WATCH_TEMPLATE_PATH):
        return static_file("watch_template.html", root=BASE_DIR)
    return "<div style='color:white;background:black;padding:20px;font-family:sans-serif;'>Watch template not found.</div>"

@app.route("/health")
@app.route("/api/health")
def health_check():
    response.content_type = "application/json"
    response.set_header("Access-Control-Allow-Origin", "*")
    return json.dumps({
        "status": "ok",
        "service": "XIAO ESP32-S3 Autonomous Cloud Solver",
        "mock_mode": MOCK_MODE,
        "ocr_model": OCR_MODEL,
        "solver_model": SOLVER_MODEL,
        "has_api_key": bool(ANTHROPIC_API_KEY),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/api/latest_solution")
def get_latest_solution():
    response.content_type = "application/json"
    response.set_header("Access-Control-Allow-Origin", "*")
    return json.dumps(latest_solution)

latest_live_frame = None
latest_live_timestamp = 0.0
live_frame_counter = 0
latest_upload_timestamp = 0.0
last_solve_trigger_time = 0.0
solve_lock = threading.Lock()

@app.route("/api/latest_photo")
def get_latest_photo():
    photo_path = os.path.join(FINAL_PHOTOS_DIR, "ideal_capture.jpg")
    if os.path.exists(photo_path):
        response.set_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        response.set_header("Access-Control-Allow-Origin", "*")
        return static_file("ideal_capture.jpg", root=FINAL_PHOTOS_DIR, mimetype="image/jpeg")
    response.status = 404
    return "No photo captured yet"

@app.route("/upload_live", method=["POST", "OPTIONS"])
def handle_upload_live():
    global latest_live_frame, latest_live_timestamp, live_frame_counter
    if request.method == "OPTIONS":
        response.set_header("Access-Control-Allow-Origin", "*")
        response.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.set_header("Access-Control-Allow-Headers", "*")
        return ""

    body = request.body.read()
    if body:
        try:
            arr = np.frombuffer(body, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                img_rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                _, enc = cv2.imencode(".jpg", img_rot, [cv2.IMWRITE_JPEG_QUALITY, 80])
                latest_live_frame = enc.tobytes()
                latest_live_timestamp = time.time()
                live_frame_counter += 1
                try:
                    with open(LIVE_FRAME_PATH, "wb") as f:
                        f.write(latest_live_frame)
                except Exception as ef:
                    print(f"[LIVE STREAM] Błąd zapisu {LIVE_FRAME_PATH}: {ef}")
        except Exception as e:
            print(f"[LIVE STREAM BŁĄD] {e}")

    response.set_header("Access-Control-Allow-Origin", "*")
    return json.dumps({"status": "ok", "frame": live_frame_counter})

@app.route("/live_frame")
@app.route("/api/live_stream")
def get_live_stream():
    global latest_live_frame
    response.set_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
    response.set_header("Access-Control-Allow-Origin", "*")
    response.content_type = "image/jpeg"
    if latest_live_frame is not None:
        return latest_live_frame
    if os.path.exists(LIVE_FRAME_PATH):
        return static_file("latest_live_frame.jpg", root=FINAL_PHOTOS_DIR, mimetype="image/jpeg")
    photo_path = os.path.join(FINAL_PHOTOS_DIR, "ideal_capture.jpg")
    if os.path.exists(photo_path):
        return static_file("ideal_capture.jpg", root=FINAL_PHOTOS_DIR, mimetype="image/jpeg")
    response.status = 404
    return "No frame"

@app.route("/api/camera_status")
def get_camera_status():
    global latest_live_timestamp, latest_live_frame, live_frame_counter, latest_upload_timestamp
    response.content_type = "application/json"
    response.set_header("Access-Control-Allow-Origin", "*")
    response.set_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
    
    now = time.time()
    is_live = (now - latest_live_timestamp < 4.0) if (latest_live_frame is not None and latest_live_timestamp > 0) else False
    is_recent_upload = (now - latest_upload_timestamp < 12.0)
    is_online = is_live or is_recent_upload

    last_photo_time = 0
    photo_path = os.path.join(FINAL_PHOTOS_DIR, "ideal_capture.jpg")
    has_photo = os.path.exists(photo_path)
    if has_photo:
        last_photo_time = os.path.getmtime(photo_path)

    return json.dumps({
        "connected": is_online,
        "is_online": is_online,
        "status": "live" if is_live else ("active" if is_recent_upload else ("cached" if has_photo else "sleeping")),
        "has_photo": has_photo,
        "last_live_seconds_ago": round(now - latest_live_timestamp, 1) if latest_live_timestamp > 0 else 9999,
        "last_upload_seconds_ago": round(now - latest_upload_timestamp, 1) if latest_upload_timestamp > 0 else 9999,
        "last_photo_seconds_ago": round(now - last_photo_time, 1) if last_photo_time > 0 else 9999,
        "frames_received": live_frame_counter
    })

@app.route("/api/clear", method=["POST", "GET", "OPTIONS"])
def handle_clear():
    global latest_solution, burst_accumulator, latest_live_frame, latest_live_timestamp, live_frame_counter
    if request.method == "OPTIONS":
        response.set_header("Access-Control-Allow-Origin", "*")
        response.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        response.set_header("Access-Control-Allow-Headers", "*")
        return ""

    burst_accumulator.clear()
    latest_live_frame = None
    latest_live_timestamp = 0.0
    live_frame_counter = 0
    now_str = datetime.now().strftime("%H:%M:%S")
    latest_solution = {
        "status": "cleared",
        "html": """
        <div class="idle-state">
          <div class="gemini-orb-wrapper">
            <div class="gemini-glow"></div>
            <div id="mainOrb" class="gemini-orb"></div>
          </div>
          <div class="idle-title">CZYSTKA WYKONANA</div>
          <div class="idle-sub">Poprzednie zadania wymazane.<br>Oczekiwanie na nowe zdjęcie z kamery.</div>
        </div>
        """,
        "time": now_str,
        "hash": "cleared_" + str(int(time.time())),
        "guidance": "",
        "mock_mode": MOCK_MODE
    }
    with open(SOLUTION_FILE, "w", encoding="utf-8") as f:
        json.dump(latest_solution, f, ensure_ascii=False, indent=2)

    print(f"[CZYSTKA] System wyczyszczony o godz. {now_str}! Oczekiwanie na nowe zdjęcie.")
    response.set_header("Access-Control-Allow-Origin", "*")
    return json.dumps({"status": "cleared", "time": now_str})

def execute_solution_pipeline(frame_bytes):
    global latest_solution
    now_str = datetime.now().strftime("%H:%M:%S")
    latest_solution["status"] = "processing"
    latest_solution["stage"] = 1
    latest_solution["stage_text"] = "1. Odbieranie klatki..."
    latest_solution["time"] = now_str
    
    with open(SOLUTION_FILE, "w", encoding="utf-8") as f:
        json.dump(latest_solution, f, ensure_ascii=False, indent=2)

    full_img, crops = enhance_and_crop(frame_bytes)
    reload_env()

    if MOCK_MODE:
        print("[MOCK] Tryb testowy: generowanie makiety bez użycia API (0 zł)...")
        latest_solution["stage"] = 2
        latest_solution["stage_text"] = "2. Claude Haiku analizuje..."
        with open(SOLUTION_FILE, "w", encoding="utf-8") as f:
            json.dump(latest_solution, f, ensure_ascii=False, indent=2)
        time.sleep(0.4)

        latest_solution["stage"] = 3
        latest_solution["stage_text"] = "3. Claude Opus rozwiązuje..."
        with open(SOLUTION_FILE, "w", encoding="utf-8") as f:
            json.dump(latest_solution, f, ensure_ascii=False, indent=2)
        time.sleep(0.5)

        mock_html = generate_mock_solution(crops, now_str)
        sol_hash = "sol_" + hashlib.md5((mock_html + str(time.time())).encode("utf-8")).hexdigest()[:8]

        latest_solution["status"] = "ready"
        latest_solution["stage"] = 4
        latest_solution["stage_text"] = "4. Gotowe!"
        latest_solution["html"] = mock_html
        latest_solution["time"] = now_str
        latest_solution["hash"] = sol_hash
        latest_solution["mock_mode"] = True
    else:
        print(f"[LIVE AI] Etap 1: Claude Haiku 5.5 audytuje jakość i transkrybuje zadania...")
        latest_solution["stage"] = 2
        latest_solution["stage_text"] = "2. Claude Haiku analizuje..."
        with open(SOLUTION_FILE, "w", encoding="utf-8") as f:
            json.dump(latest_solution, f, ensure_ascii=False, indent=2)

        audit = run_stage_1_auditor(crops)
        legible = audit.get("legible_tasks", []) if audit else []
        unclear = audit.get("unclear_tasks", []) if audit else []
        guidance = audit.get("camera_instruction", "") if audit else ""

        print(f"[LIVE AI] Etap 2: Claude Opus 5.5 rozwiązuje {len(legible)} czytelnych zadań...")
        latest_solution["stage"] = 3
        latest_solution["stage_text"] = "3. Claude Opus rozwiązuje..."
        with open(SOLUTION_FILE, "w", encoding="utf-8") as f:
            json.dump(latest_solution, f, ensure_ascii=False, indent=2)

        solved_html = run_stage_2_opus(legible) if legible else "<div style='color:#ef4444; padding: 20px;'>Nie wykryto czytelnych zadań na zdjęciu. Popraw kadr według wskazówki.</div>"
        banner_html = build_guidance_banner(unclear, guidance)
        final_html = banner_html + solved_html
        sol_hash = "sol_" + hashlib.md5((final_html + str(time.time())).encode("utf-8")).hexdigest()[:8]

        latest_solution["status"] = "ready"
        latest_solution["stage"] = 4
        latest_solution["stage_text"] = "4. Gotowe!"
        latest_solution["html"] = final_html
        latest_solution["time"] = now_str
        latest_solution["hash"] = sol_hash
        latest_solution["guidance"] = guidance
        latest_solution["mock_mode"] = False

    with open(SOLUTION_FILE, "w", encoding="utf-8") as f:
        json.dump(latest_solution, f, ensure_ascii=False, indent=2)

    print(f"[SOLVER] Rozwiązanie opublikowane na zegarku (hash: {sol_hash})!")
    return sol_hash

@app.route("/api/trigger_solve", method=["POST", "GET", "OPTIONS"])
@app.route("/api/solve_live", method=["POST", "GET", "OPTIONS"])
def handle_solve_live():
    global latest_live_frame, latest_live_timestamp, latest_solution, last_solve_trigger_time
    if request.method == "OPTIONS":
        response.set_header("Access-Control-Allow-Origin", "*")
        response.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        response.set_header("Access-Control-Allow-Headers", "*")
        return ""

    response.set_header("Access-Control-Allow-Origin", "*")
    response.content_type = "application/json"

    with solve_lock:
        now = time.time()
        # 1. OCHRONA DEBOUNCE: Blokada ponownych wywołań w oknie 5 sekund
        elapsed_since_last = now - last_solve_trigger_time
        if elapsed_since_last < 5.0:
            remaining = round(5.0 - elapsed_since_last, 1)
            print(f"[DEBOUNCE] Zablokowano zbyt częste wywołanie API ({remaining}s do odblokowania)!")
            response.status = 429
            return json.dumps({
                "status": "error",
                "reason": "debounced",
                "message": f"Ochrona debounce: odczekaj {remaining}s przed kolejnym wywołaniem.",
                "retry_after_seconds": remaining
            })

        # 2. OCHRONA PRZED RÓWNOLEGŁYM PRZETWARZANIEM
        if latest_solution.get("status") == "processing":
            print(f"[CONCURRENCY] Zablokowano wywołanie - poprzednia analiza AI w toku")
            response.status = 429
            return json.dumps({
                "status": "error",
                "reason": "already_processing",
                "message": "Analiza AI jest już w toku. Poczekaj na ukończenie poprzedniego zadania."
            })

        # 3. WALIDACJA ŚWIEŻOŚCI KLATKI PODGLĄDU
        allow_cached = request.query.get("allow_cached", "").lower() in ("true", "1")
        frame_to_use = None
        frame_source = "none"
        live_age = None

        if latest_live_frame is not None and latest_live_timestamp > 0:
            live_age = now - latest_live_timestamp
            if live_age <= 8.0:
                frame_to_use = latest_live_frame
                frame_source = f"live_stream ({round(live_age, 2)}s)"

        if frame_to_use is None and os.path.exists(LIVE_FRAME_PATH):
            file_mtime = os.path.getmtime(LIVE_FRAME_PATH)
            file_age = now - file_mtime
            if file_age <= 8.0:
                try:
                    with open(LIVE_FRAME_PATH, "rb") as f:
                        frame_to_use = f.read()
                    frame_source = f"live_file ({round(file_age, 2)}s)"
                    live_age = file_age
                except Exception as ef:
                    print(f"[SOLVE LIVE] Błąd odczytu {LIVE_FRAME_PATH}: {ef}")

        # Jeśli brak świeżej klatki live, sprawdź tryb awaryjny (allow_cached np. dla zapisanego zdjęcia)
        if frame_to_use is None and (allow_cached or MOCK_MODE):
            photo_path = os.path.join(FINAL_PHOTOS_DIR, "ideal_capture.jpg")
            if os.path.exists(photo_path):
                try:
                    with open(photo_path, "rb") as f:
                        frame_to_use = f.read()
                    frame_source = "cached_photo (ideal_capture.jpg)"
                except Exception as ef:
                    print(f"[SOLVE LIVE] Błąd odczytu {photo_path}: {ef}")
            elif MOCK_MODE:
                # W trybie mock na świeżym kontenerze generujemy syntetyczną klatkę testową
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(dummy, "MOCK TEST CARD", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (56, 189, 248), 2)
                _, enc = cv2.imencode(".jpg", dummy)
                frame_to_use = enc.tobytes()
                frame_source = "synthetic_mock_card"

        # Jeśli brak świeżego kadru - odrzuć bez wywoływania API
        if frame_to_use is None:
            age_info = f" (ostatnia klatka sprzed {round(live_age, 1)}s, limit: 8s)" if live_age is not None else " (brak klatek w buforze)"
            msg = f"Kamera jest w uśpieniu lub offline{age_info}. Włącz podgląd na żywo, aby rozwiązać bieżący kadr."
            print(f"[VALIDATION FAIL] {msg}")
            response.status = 400
            return json.dumps({
                "status": "error",
                "reason": "stale_or_missing_frame",
                "message": msg,
                "live_age_seconds": round(live_age, 1) if live_age is not None else None
            })

        # Zarejestruj czas wywołania dla debounce
        last_solve_trigger_time = now

    now_str = datetime.now().strftime("%H:%M:%S")
    print(f"\n[SOLVE TRIGGER] Uruchomienie rozwiązania ({frame_source}) o godz. {now_str}...")

    latest_solution["status"] = "processing"
    latest_solution["stage"] = 1
    latest_solution["stage_text"] = "1. Odbieranie klatki..."
    latest_solution["time"] = now_str
    latest_solution["hash"] = "proc_" + str(int(time.time()))
    with open(SOLUTION_FILE, "w", encoding="utf-8") as f:
        json.dump(latest_solution, f, ensure_ascii=False, indent=2)

    frame_copy = bytes(frame_to_use)
    ai_executor.submit(execute_solution_pipeline, frame_copy)

    return json.dumps({
        "status": "processing",
        "stage": 1,
        "stage_text": "1. Odbieranie klatki...",
        "time": now_str,
        "frame_source": frame_source
    })

@app.route("/upload", method=["POST", "OPTIONS"])
def handle_upload():
    global latest_solution
    if request.method == "OPTIONS":
        response.set_header("Access-Control-Allow-Origin", "*")
        response.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.set_header("Access-Control-Allow-Headers", "*")
        return ""

    body = request.body.read()
    if not body:
        response.status = 400
        return json.dumps({"status": "error", "message": "Empty body"})

    burst_id = request.headers.get("X-Burst-Id", str(int(time.time())))
    burst_idx = int(request.headers.get("X-Burst-Index", "1"))
    burst_total = int(request.headers.get("X-Burst-Total", "1"))

    print(f"[UPLOAD] Odebrano klatkę {burst_idx}/{burst_total} (Burst ID: {burst_id}, {len(body)} B)")

    if burst_id not in burst_accumulator:
        burst_accumulator[burst_id] = []

    burst_accumulator[burst_id].append(body)

    if len(burst_accumulator[burst_id]) >= burst_total or burst_total == 1:
        frames = burst_accumulator.pop(burst_id)
        now_str = datetime.now().strftime("%H:%M:%S")
        print(f"[PIPELINE] Przetwarzanie serii {len(frames)} klatek o godz. {now_str} (wątek asynchroniczny)...")

        best_frame = select_sharpest_frame(frames) if len(frames) > 1 else frames[0]

        # Natychmiast ustawiamy status "processing"
        global latest_upload_timestamp
        latest_upload_timestamp = time.time()
        latest_solution["status"] = "processing"
        latest_solution["stage"] = 1
        latest_solution["stage_text"] = "1. Odbieranie klatki..."
        latest_solution["time"] = now_str
        latest_solution["hash"] = "proc_" + str(int(time.time()))
        with open(SOLUTION_FILE, "w", encoding="utf-8") as f:
            json.dump(latest_solution, f, ensure_ascii=False, indent=2)

        ai_executor.submit(execute_solution_pipeline, best_frame)

    response.set_header("Access-Control-Allow-Origin", "*")
    return json.dumps({"status": "processing", "burst_id": burst_id, "index": burst_idx, "mock": MOCK_MODE})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"\n==========================================================")
    print(f"  SERWER CLOUD SOLVER (WIELOWĄTKOWY - ASYNCHRONICZNY)")
    print(f"  Tryb: {'MOCK (0 ZŁ)' if MOCK_MODE else 'API LIVE'}")
    print(f"  Host: {host} | Port: {port} | UI: http://{host}:{port}/watch")
    print(f"==========================================================\n")
    serve(app, host=host, port=port, threads=8)
