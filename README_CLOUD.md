# 🚀 24/7 Autonomiczny Cloud Deployment: XIAO ESP32-S3 Exam Solver

Kompletny pakiet wdrożeniowy backendu AI i interfejsu smartwatcha dla **Seeed Studio XIAO ESP32-S3 Sense**.

> **Cel systemu:** 100% autonomia w terenie bez konieczności włączania laptopa!  
> **Architektura:** `Smartwatch (AMOLED/KaTeX)` $\longleftrightarrow$ `Chmura HTTPS (Render/HF Spaces)` $\longleftrightarrow$ `ESP32-S3 Sense (Kamera + Hotspot z telefonu)`.

---

## 📁 Zawartość paczki wdrożeniowej

| Plik | Rola w systemie |
| :--- | :--- |
| `cloud_solver.py` | Asynchroniczny silnik backendu (Waitress + Bottle, pipeline OCR Haiku + Solver Opus + Mock) |
| `watch_template.html` | Zoptymalizowany interfejs smartwatcha (czarne tło AMOLED #000000, KaTeX, SVG) |
| `Dockerfile` | Multi-stage build na Pythonie 3.11 z `libgl1`, `libglib2.0`, OpenCV, Waitress i cURL |
| `render.yaml` | Blueprint 1-Click do natychmiastowego wdrożenia na darmowym tierze Render.com |
| `Procfile` | Deklaracja procesu dla platform PaaS (`web: python cloud_solver.py`) |
| `requirements.txt` | Czyste, minimalne i przypięte zależności produkcyjne |
| `.env.example` | Szablon konfiguracji zmiennych środowiskowych |
| `.dockerignore` | Wykluczenie zbędnych plików z obrazu kontenera |

---

## ⚡ Opcja 1: Wdrożenie na Render.com (100% Darmowe, 2 minuty)

Platforma **Render.com** oferuje w 100% darmowy hosting Web Service z certyfikatem HTTPS.

### Krok 1: Wypchnięcie kodu na GitHub
1. Utwórz nowe repozytorium na swoim koncie GitHub, np. `xiao-cloud-solver` (może być prywatne).
2. Skopiuj zawartość folderu `cloud_deploy` i wykonaj:
   ```bash
   git init
   git add .
   git commit -m "Initial cloud deployment release"
   git branch -M main
   git remote add origin https://github.com/<twoj-login>/xiao-cloud-solver.git
   git push -u origin main
   ```

### Krok 2: Wdrożenie 1-Click przez Blueprint
1. Zaloguj się na [Render.com](https://render.com).
2. W prawym górnym rogu kliknij **New +** $\rightarrow$ **Blueprint**.
3. Połącz swoje konto GitHub i wskaż repozytorium `xiao-cloud-solver`.
4. Render automatycznie odczyta plik `render.yaml`:
   - Ustawi plan: **Free**
   - Środowisko: **Docker**
   - Healthcheck: `/health`
5. Jeśli chcesz od razu korzystać z płatnego API Claude (zamiast darmowej makiety MOCK):
   - W sekcji zmiennych podaj swój `ANTHROPIC_API_KEY`.
   - Zmień `MOCK_MODE` na `false`.
6. Kliknij **Apply**. 

Po ok. 90 sekundach kontener zostanie zbudowany i uruchomiony pod stałym adresem HTTPS:
```
https://xiao-solver-backend-xxxx.onrender.com
```

### 💡 Wskazówka: Utrzymanie kontenera w stanie czuwania (Anti-Sleep)
Darmowe serwisy na Render zasypiają po 15 minutach bezczynności. Aby serwer odpowiadał natychmiast 24/7:
1. Zarejestruj darmowe konto na [cron-job.org](https://cron-job.org) lub [uptimerobot.com](https://uptimerobot.com).
2. Skonfiguruj pingowanie adresu:
   `https://twoj-serwis.onrender.com/health` co **10 minut**.
3. Serwer nigdy nie przejdzie w uśpienie i będzie w 100% gotowy w każdej chwili!

---

## 🤗 Opcja 2: Wdrożenie na Hugging Face Spaces (Darmowy Docker 24/7)

Hugging Face Spaces udostępnia darmowe kontenery Docker, które **nie zasypiają** podczas sesji:

1. Zaloguj się na [Hugging Face](https://huggingface.co).
2. Kliknij swój awatar $\rightarrow$ **New Space**.
3. Nazwij Space, np. `xiao-solver`.
4. Wybierz:
   - **Space SDK:** `Docker`
   - **Docker template:** `Blank`
   - **Space hardware:** `Free (CPU basic · 2 vCPU · 16 GB)`
5. Sklonuj repozytorium Space lub wypchnij pliki z `cloud_deploy`:
   ```bash
   git remote add space https://huggingface.co/spaces/<twoj-login>/xiao-solver
   git push space main
   ```
6. W zakładce **Settings** $\rightarrow$ **Variables and secrets**:
   - Dodaj New secret: `ANTHROPIC_API_KEY` (Twój klucz Anthropic)
   - Dodaj New variable: `MOCK_MODE` = `false` (lub `true` do testów)
7. Adres aplikacji będzie dostępny w formacie:
   `https://<twoj-login>-xiao-solver.hf.space/watch`

---

## 🛠️ Konfiguracja ESP32-S3 Sense (Firmware `src/main.cpp`)

Gdy Twój serwer działa już w chmurze, nie potrzebujesz uruchamiać laptopa ani lokalnego serwera!

W pliku firmware `src/main.cpp` na ESP32-S3 zaktualizuj adresy URL:

```cpp
// 1. Konfiguracja mobilnego Hotspotu w smartfonie
const char* ssid     = "MojaNazwaHotspotu";
const char* password = "HasloDoHotspotu";

// 2. Stały adres serwera w chmurze HTTPS (Render lub Hugging Face)
const char* server_upload_url = "https://twoj-serwis.onrender.com/upload";
const char* server_stream_url = "https://twoj-serwis.onrender.com/upload_live";
```

Wgraj firmware do płytki XIAO ESP32-S3 Sense raz przez kabel USB. Od tego momentu płytka jest w pełni autonomiczna!

---

## ⌚ Obsługa ze Smartwatcha

1. Włącz Hotspot w telefonie (z nim łączy się ESP32-S3).
2. Na smartwatchu (np. Galaxy Watch, Apple Watch, Pixel Watch lub dowolnym zegarku z Wear OS / przeglądarką) otwórz przeglądarkę internetową i wejdź na:
   ```
   https://twoj-serwis.onrender.com/watch
   ```
3. Zegarek wyświetli dopasowany do okrągłego ekranu interfejs z:
   - Paskiem stanu kamery na żywo (Online / Offline / Sleeping).
   - Przyciskami `Rozwiąż`, `Wyczyść`, `Podgląd kadru`.
   - Pełnym renderowaniem wzorów w **KaTeX** i schematami wektorowymi **SVG**.

---

## 🧪 Testowanie lokalne w Dockerze (Opcjonalne)

Przed wdrożeniem do chmury możesz przetestować kontener lokalnie:

```bash
cd cloud_deploy
docker build -t xiao-cloud-solver .
docker run -it -p 5000:5000 -e MOCK_MODE=true xiao-cloud-solver
```
Otwórz w przeglądarce: [http://localhost:5000/watch](http://localhost:5000/watch)
