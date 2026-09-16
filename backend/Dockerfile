FROM python:3.11-slim

WORKDIR /app

# System deps for MNE (BLAS) and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libsndfile1 libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-configure MNE data dir and create weights dir
RUN mkdir -p /root/mne_data weights && \
    python -c "import mne; mne.set_config('MNE_DATA', '/root/mne_data', set_env=True); mne.set_log_level('ERROR')"

EXPOSE 8000

# Run pre-flight check then start server
CMD ["sh", "-c", "python startup_check.py && uvicorn main:app --host 0.0.0.0 --port 8000 --log-level warning"]
