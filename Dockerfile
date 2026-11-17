# =========================================================================
# ETAP 1: Budowniczy (Kompilacja i pobieranie paczek kółkowych)
# =========================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Zmienne środowiskowe dla pip
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalacja podstawowych narzędzi kompilacji jeśli wymagane
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# =========================================================================
# ETAP 2: Środowisko uruchomieniowe (Ultra-lekki kontener produkcyjny)
# =========================================================================
FROM python:3.11-slim as runner

WORKDIR /app

# Instalacja bibliotek systemowych wymaganych przez OpenCV i curl dla healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Kopiowanie zainstalowanych bibliotek z etapu pierwszego
COPY --from=builder /install /usr/local

# Kopiowanie kodu aplikacji
COPY . /app

# Utworzenie katalogów na zdjęcia jeśli nie istnieją
RUN mkdir -p /app/final_photos /app/captured_photos

# Konfiguracja środowiska
ENV PYTHONUNBUFFERED=1 \
    PORT=5000 \
    HOST=0.0.0.0

EXPOSE 5000

# Sprawdzanie kondycji kontenera przez Render / Docker
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT}/health || exit 1

# Uruchomienie serwera
CMD ["python", "cloud_solver.py"]
