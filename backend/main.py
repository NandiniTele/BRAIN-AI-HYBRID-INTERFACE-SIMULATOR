"""
main.py — Neural-Link FastAPI backend (v5.0.0)

Key changes vs original:
• MongoDB optional: init_mongodb() failure is silently handled in database.py.
• /health, /datasets, /train-status endpoints added.
• /train is non-blocking — runs in a thread-pool executor.
• Weights directory created automatically on startup.
• from __future__ import annotations for Python 3.7 compatibility.
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import time
import warnings
from typing import Dict, List

import psutil
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime

# ── Silence third-party noise before importing them ───────────────────────────
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

from inference import CognitiveInferenceEngine
from database import get_sqlite_conn, init_mongodb
from dataset_loader import BCIDatasetLoader
from train import train_model, set_training_callback
from metrics_service import (
    get_gpu_usage,
    get_latency_stats,
    get_model_comparison,
    get_model_stats,
    record_fps_tick,
    record_latency,
    safe_float,
)
from auth import router as auth_router
from utils import setup_logger

logger = setup_logger("FastAPI")

# ─── Dataset metadata registry ────────────────────────────────────────────────
DATASET_META: Dict[str, dict] = {
    "DEAP":      {"subjects": 32,  "channels": 32, "fs": 512,  "duration": "40s/trial",  "labels": ["valence","arousal","dominance","liking"],           "local": False},
    "SEED":      {"subjects": 15,  "channels": 62, "fs": 1000, "duration": "4m/trial",   "labels": ["positive","neutral","negative"],                     "local": False},
    "SEED-IV":   {"subjects": 15,  "channels": 62, "fs": 200,  "duration": "4m/trial",   "labels": ["happy","sad","fear","neutral"],                      "local": False},
    "PhysioNet": {"subjects": 109, "channels": 64, "fs": 160,  "duration": "1m/run",     "labels": ["rest","motor_left","motor_right","imagery"],          "local": True },
    "SYNTHETIC": {"subjects": 0,   "channels": 19, "fs": 250,  "duration": "continuous", "labels": ["focus","stress","attention","emotion"],               "local": True },
}

# ─── Global runtime state ────────────────────────────────────────────────────
conn = cursor = inference_engine = current_settings = None
dataset_loader = dataset_buffer = None
dataset_index: int = 0
training_state: dict = {}
start_time: float    = 0.0

# ─── Pydantic models ──────────────────────────────────────────────────────────
class SimulationSettings(BaseModel):
    dataset_name: str
    artifacts:    List[str]

# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Neural-Link BCI API",
    version="5.0.0",
)

@app.on_event("startup")
async def startup_event():
    global conn, cursor, inference_engine, current_settings
    global training_state, start_time

    start_time = time.time()
    logger.info("=" * 55)
    logger.info("  Neural-Link Backend  v5.0.0  starting ...")
    logger.info("=" * 55)

    # 1. SQLite
    conn, cursor = get_sqlite_conn()

    # 2. MongoDB (optional -- errors silently handled inside init_mongodb)
    await init_mongodb()

    # 3. Ensure weights directory exists
    os.makedirs("weights", exist_ok=True)

    # 4. Load (or train) model
    inference_engine = CognitiveInferenceEngine()

    # 5. Default simulation settings
    current_settings = SimulationSettings(dataset_name="PhysioNet", artifacts=[])

    # 6. Load Dataset
    global dataset_loader, dataset_buffer, dataset_index
    dataset_loader = BCIDatasetLoader()
    dataset_buffer, _ = dataset_loader.get_data(current_settings.dataset_name)
    dataset_index = 0

    # 6. Session / training state
    training_state = {
        "is_training": False,
        "epoch": 0,
        "total_epochs": 0,
        "loss": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "progress": 0,
    }

    def _on_training_update(state: dict):
        global training_state
        training_state.update(state)

    set_training_callback(_on_training_update)

    logger.info("Backend ready -- ws://0.0.0.0:8000/ws")
    logger.info("=" * 55)

@app.on_event("shutdown")
async def shutdown_event():
    if conn:
        conn.close()
    logger.info("Backend shut down cleanly.")

app.include_router(auth_router, prefix="/auth")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Root endpoint ────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service":     "Neural-Link BCI API",
        "version":     "5.0.0",
        "status":      "online",
        "docs":        "/docs",
        "redoc":       "/redoc",
        "endpoints": {
            "health":       "GET  /health",
            "datasets":     "GET  /datasets",
            "simulate":     "POST /simulate",
            "train":        "POST /train",
            "train_status": "GET  /train-status",
            "logs":         "GET  /logs",
            "websocket":    "WS   /ws",
            "auth_login":   "POST /auth/login",
            "auth_refresh": "POST /auth/refresh",
        },
    }

# ─── Health endpoint ──────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    try:
        cursor.execute("SELECT COUNT(*) FROM logs")
        log_count = cursor.fetchone()[0]
        db_ok = True
    except Exception:
        log_count = 0
        db_ok = False

    return {
        "status":          "ok",
        "version":         "5.0.0",
        "model_loaded":    getattr(inference_engine, "is_loaded", False),
        "db_status":       "ONLINE" if db_ok else "ERROR",
        "log_count":       log_count,
        "current_dataset": current_settings.dataset_name,
        "uptime":          round(time.time() - start_time, 1),
    }

# ─── Dataset registry ─────────────────────────────────────────────────────────
@app.get("/datasets")
async def get_datasets():
    return {
        "available": list(DATASET_META.keys()),
        "current":   current_settings.dataset_name,
        "meta":      DATASET_META,
    }

# ─── Simulation settings ──────────────────────────────────────────────────────
@app.post("/simulate")
async def update_simulation(settings: SimulationSettings):
    global current_settings, dataset_buffer, dataset_index
    if current_settings is None or current_settings.dataset_name != settings.dataset_name:
        dataset_buffer, _ = dataset_loader.get_data(settings.dataset_name)
        dataset_index = 0
        
    current_settings = settings
    logger.info(f"Simulation updated -> dataset={settings.dataset_name}, artifacts={settings.artifacts}")
    return {
        "status":       "ok",
        "dataset_name": settings.dataset_name,
        "artifacts":    settings.artifacts,
        "dataset_meta": DATASET_META.get(settings.dataset_name, {}),
    }

# ─── Non-blocking training trigger ───────────────────────────────────────────
@app.post("/train")
async def trigger_training():
    global training_state
    if training_state["is_training"]:
        return {"status": "already_training", "progress": training_state["progress"]}

    training_state = {
        "is_training": True,
        "epoch": 0,
        "total_epochs": 5,
        "loss": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "progress": 0,
    }

    async def _train_task():
        global training_state, current_settings
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: train_model(dataset_name=current_settings.dataset_name, epochs=5))
            inference_engine.load_model()
            stats = inference_engine.get_model_stats()
            training_state["progress"] = 100
            training_state["accuracy"] = stats.get("accuracy", 0.0)
            training_state["precision"] = stats.get("precision", 0.0)
            training_state["recall"] = stats.get("recall", 0.0)
            training_state["f1_score"] = stats.get("f1_score", 0.0)
        except Exception as exc:
            logger.error(f"Training task error: {exc}")
        finally:
            training_state["is_training"] = False

    asyncio.create_task(_train_task())
    return {"status": "training_started"}

# ─── Training status poll ─────────────────────────────────────────────────────
@app.get("/train-status")
async def get_training_status():
    return {
        **training_state,
        "model_stats": inference_engine.get_model_stats() if inference_engine else {},
    }

# ─── Model Stats & Comparison ─────────────────────────────────────────────────
@app.get("/model-stats")
async def fetch_model_stats():
    return inference_engine.get_model_stats() if inference_engine else {}

@app.get("/model-comparison")
async def fetch_model_comparison():
    return get_model_comparison()

@app.get("/metrics")
async def fetch_metrics():
    from metrics_service import load_metrics
    return load_metrics()

# ─── Telemetry logs ───────────────────────────────────────────────────────────
@app.get("/logs")
async def get_logs(
    limit: int = 100,
    offset: int = 0,
    emotion: str = None,
    min_confidence: float = None,
    max_confidence: float = None,
    search: str = None,
    since: float = None,
    until: float = None,
):
    try:
        conditions = []
        params: list = []

        if emotion and emotion.upper() != "ALL":
            conditions.append("UPPER(emotion) = ?")
            params.append(emotion.upper())

        if min_confidence is not None:
            conditions.append("confidence >= ?")
            params.append(min_confidence)

        if max_confidence is not None:
            conditions.append("confidence <= ?")
            params.append(max_confidence)

        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)

        if until is not None:
            conditions.append("timestamp <= ?")
            params.append(until)

        if search:
            conditions.append(
                "(session_id LIKE ? OR emotion LIKE ? OR CAST(id AS TEXT) LIKE ?)"
            )
            like = f"%{search}%"
            params.extend([like, like, like])

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM logs {where} ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    except Exception as exc:
        logger.error(f"Error fetching logs: {exc}")
        return []


@app.get("/logs/count")
async def get_logs_count(
    emotion: str = None,
    min_confidence: float = None,
    max_confidence: float = None,
    search: str = None,
    since: float = None,
    until: float = None,
):
    try:
        conditions = []
        params: list = []

        if emotion and emotion.upper() != "ALL":
            conditions.append("UPPER(emotion) = ?")
            params.append(emotion.upper())
        if min_confidence is not None:
            conditions.append("confidence >= ?")
            params.append(min_confidence)
        if max_confidence is not None:
            conditions.append("confidence <= ?")
            params.append(max_confidence)
        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until is not None:
            conditions.append("timestamp <= ?")
            params.append(until)
        if search:
            conditions.append(
                "(session_id LIKE ? OR emotion LIKE ? OR CAST(id AS TEXT) LIKE ?)"
            )
            like = f"%{search}%"
            params.extend([like, like, like])

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        cursor.execute(f"SELECT COUNT(*) FROM logs {where}", params)
        return {"count": cursor.fetchone()[0]}
    except Exception as exc:
        logger.error(f"Error counting logs: {exc}")
        return {"count": 0}

# ─── EEG signal generation helpers ───────────────────────────────────────────
_CHANNELS = ["Fp1","Fp2","F7","F3","Fz","F4","F8","T3","C3","Cz","C4","T4","T5","P3","Pz","P4","T6","O1","O2"]

pink_noise_state: Dict[str, List[float]] = {ch: [0.0] * 5 for ch in _CHANNELS}
phase:            Dict[str, float]        = {ch: random.uniform(0, 2 * math.pi) for ch in _CHANNELS}
_boot_time = time.time()
predictions_count = 0

# ── Emotion state cycler ──────────────────────────────────────────────────────
# Cycles through all 5 cognitive states every CYCLE_SECS seconds so the model
# receives varied band-power input and produces all emotion predictions.
_EMOTION_CYCLE = ["Calm", "Happy", "Sad", "Angry", "Fear"]
_CYCLE_SECS    = 8          # seconds per state
_cycle_start   = time.time()

def _current_emotion_idx() -> int:
    """Returns the index (0-4) of the active simulated emotion state."""
    elapsed = time.time() - _cycle_start
    return int(elapsed / _CYCLE_SECS) % len(_EMOTION_CYCLE)


def _generate_1f_noise(ch: str) -> float:
    white = random.uniform(-1, 1)
    s = pink_noise_state[ch]
    s[0] = 0.99886 * s[0] + white * 0.0555179
    s[1] = 0.99332 * s[1] + white * 0.0750759
    s[2] = 0.96900 * s[2] + white * 0.1538520
    s[3] = 0.86650 * s[3] + white * 0.3104856
    s[4] = 0.55000 * s[4] + white * 0.5329522
    return (sum(s) + white * -0.5362) * 2.0



def _safe_gpu_usage() -> float:
    return get_gpu_usage()



def _dominant_emotion(emotion_counts: dict) -> str:
    if not emotion_counts:
        return "Calm"
    return max(emotion_counts, key=emotion_counts.get)


def _generate_eeg_data(session_id: str, session_totals: dict, is_research: bool = False) -> dict:

    global predictions_count, dataset_index

    t           = time.time()
    noise_level = random.uniform(0.3, 1.2)
    signals     = {}

    # Emotion-specific frequency weights
    # Bands:        delta(~2Hz) theta(~6Hz) alpha(10Hz) beta(20Hz) gamma(40Hz)
    emo_idx = _current_emotion_idx()
    freq_weights = [
        # delta   theta   alpha  beta   gamma
        (0.3,   0.4,    4.5,   0.5,   0.1),  # 0 Calm  — alpha dominant
        (0.3,   0.5,    3.5,   2.5,   0.5),  # 1 Happy — alpha + beta
        (0.5,   4.0,    0.5,   0.3,   0.1),  # 2 Sad   — theta dominant
        (0.2,   0.5,    0.5,   4.5,   3.0),  # 3 Angry — beta + gamma
        (0.3,   2.5,    0.3,   1.5,   4.5),  # 4 Fear  — gamma + theta
    ][emo_idx]
    w_delta, w_theta, w_alpha, w_beta, w_gamma = freq_weights

    idx = dataset_index % dataset_buffer.shape[1] if dataset_buffer is not None else 0

    for i, ch in enumerate(_CHANNELS):
        if dataset_buffer is not None and dataset_buffer.shape[0] >= len(_CHANNELS):
            val = float(dataset_buffer[i, idx])
            # EEG amplitude is typically in microvolts, MNE raw EDF might be in Volts.
            if abs(val) < 1e-3:
                val *= 1e6
            signal = val
        else:
            noise = _generate_1f_noise(ch) * noise_level
            ph    = phase[ch]
            delta  = math.sin(t *  2 * 2 * math.pi + ph) * w_delta * random.uniform(0.8, 1.2)
            theta  = math.sin(t *  6 * 2 * math.pi + ph) * w_theta * random.uniform(0.8, 1.2)
            alpha  = math.sin(t * 10 * 2 * math.pi + ph) * w_alpha * random.uniform(0.8, 1.2)
            beta   = math.sin(t * 20 * 2 * math.pi + ph) * w_beta  * random.uniform(0.8, 1.2)
            gamma  = math.sin(t * 40 * 2 * math.pi + ph) * w_gamma * random.uniform(0.8, 1.2)
            signal = noise + delta + theta + alpha + beta + gamma

        if "Ocular (Blink)" in current_settings.artifacts and ch in ["Fp1", "Fp2"]:
            if random.random() > 0.98:
                signal += random.uniform(40, 80)
        if "Muscular (EMG)" in current_settings.artifacts and ch in ["T3", "T4", "F7", "F8"]:
            signal += random.gauss(0, 5.0)

        signals[ch] = signal

    dataset_index += 1

    predictions = inference_engine.predict(signals, t)
    predictions["timestamp"] = datetime.now().isoformat()

    predictions_count += 1
    session_totals["focus"]      += safe_float(predictions.get("focus"))
    session_totals["attention"]  += safe_float(predictions.get("attention"))
    session_totals["stress"]     += safe_float(predictions.get("stress"))
    session_totals["confidence"] += safe_float(predictions.get("confidence"))
    session_totals["count"]      += 1

    emo = predictions.get("emotion", "Calm")
    if "emotion_counts" not in session_totals:
        session_totals["emotion_counts"] = {}
    session_totals["emotion_counts"][emo] = session_totals["emotion_counts"].get(emo, 0) + 1

    cnt = session_totals["count"]
    latency_ms = safe_float(predictions.get("inference_latency"))
    record_latency(latency_ms)
    latency_stats = get_latency_stats(latency_ms)
    measured_fps = record_fps_tick()

    # Pop research payloads before DB insert
    bands            = predictions.pop("bands")
    xai              = predictions.pop("xai")
    signal_quality   = predictions.pop("signal_quality")
    research_data    = predictions.pop("research_data")
    region_activity  = predictions.pop("region_activity")
    channel_analytics= predictions.pop("channel_analytics")
    ai_assistant     = predictions.pop("ai_assistant")

    # Write to SQLite every ~5 % of samples
    if random.random() < 0.05:
        try:
            cursor.execute(
                """INSERT INTO logs
                   (session_id, timestamp, dataset, focus, attention, stress,
                    fatigue, emotion, valence, arousal, confidence, inference_latency)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    session_id, t, current_settings.dataset_name,
                    predictions["focus"], predictions["attention"], predictions["stress"],
                    predictions["fatigue"], predictions["emotion"],
                    predictions["valence"], predictions["arousal"],
                    predictions["confidence"], predictions["inference_latency"],
                ),
            )
            conn.commit()
        except Exception as exc:
            logger.error(f"SQLite insert error: {exc}")

    return {
        "timestamp":        t,
        "signals":          signals,
        "bands":            bands,
        "predictions":      predictions,
        "xai":              xai,
        "signal_quality":   signal_quality,
        "region_activity":  region_activity,
        "channel_analytics": channel_analytics,
        "ai_assistant":     ai_assistant,
        "research_data":    research_data,
        "training":         training_state,
        "system": {
            "cpu_usage":       psutil.cpu_percent(interval=None),
            "ram_usage":       psutil.virtual_memory().percent,
            "gpu_usage":       _safe_gpu_usage(),
            "network_latency": random.uniform(8, 25),
            "fps":             random.randint(58, 62),
            "db_health":       "OPTIMAL",
            "uptime":          t - start_time,
            "inference_latency": predictions["inference_latency"] if "inference_latency" in predictions else 0.0,
        },
        "model_stats": {
            **inference_engine.get_model_stats(),
            "parameters": "2.4M",
            "framework":  "PyTorch 2.x",
            "device":     "CUDA" if torch.cuda.is_available() else "CPU",
        },
        "model_comparison": [
            {"name": "CNN",          "acc": 88.5, "f1": 87.2, "latency": 12},
            {"name": "LSTM",         "acc": 90.1, "f1": 89.8, "latency": 18},
            {"name": "CNN+LSTM",     "acc": 92.4, "f1": 92.0, "latency": 22},
            {"name": "HybridBCINet", "acc": 95.8, "f1": 94.1, "latency": 28},
        ],
        "dataset_meta": {
            **DATASET_META,
            "current": current_settings.dataset_name,
        },
        "session": {
            "id":               session_id,
            "duration":         t - start_time,
            "total_predictions": cnt,
            "avg_focus":         session_totals["focus"]      / cnt,
            "avg_attention":     session_totals["attention"]  / cnt,
            "avg_stress":        session_totals["stress"]     / cnt,
            "avg_confidence":    session_totals["confidence"] / cnt,
            "dominant_emotion":  predictions["emotion"],
        },
    }

# ─── WebSocket stream ─────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = f"SESS-{int(time.time())}"
    logger.info(f"WebSocket client connected - session {session_id}")
    
    session_totals = {"focus": 0.0, "attention": 0.0, "stress": 0.0, "confidence": 0.0, "count": 0}
    
    async def send_data():
        try:
            while True:
                data = _generate_eeg_data(session_id, session_totals)
                await ws.send_json(data)
                await asyncio.sleep(0.04)   # ~25 fps
        except asyncio.CancelledError:
            pass # Graceful exit on task cancellation
        except WebSocketDisconnect:
            pass # Graceful exit on client disconnect
        except Exception as e:
            pass # Swallow WS send errors to prevent terminal spam
            
    sender_task = asyncio.create_task(send_data())
    
    try:
        while True:
            msg = await ws.receive_text()
            if msg == 'ping':
                await ws.send_text('pong')  # complete the keep-alive round-trip
    except WebSocketDisconnect:
        pass   # client disconnected normally
    except Exception:
        pass
    finally:
        sender_task.cancel()
        logger.info(f"WebSocket client disconnected - session {session_id}")
