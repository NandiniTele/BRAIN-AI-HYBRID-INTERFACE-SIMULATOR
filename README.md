# NEURAL-LINK: Advanced Brain-Computer Interface (BCI) Research Platform

![Neural-Link Architecture](https://img.shields.io/badge/Architecture-FastAPI%20%2B%20React-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)

Neural-Link is a professional-grade, full-stack Brain–Computer Interface (BCI) platform designed for neuroscience research, IEEE demonstrations, and academic projects. It integrates a real-time EEG signal processing pipeline with a deep learning inference engine (Hybrid CNN-LSTM-Transformer) and a high-fidelity cyberpunk dashboard.

## 🚀 Features

- **Real-time EEG Processing**: MNE-Python based pipeline (Band-pass 0.5-45Hz, Notch 50Hz, ICA artifact removal, Z-score normalization).
- **Deep Learning Inference**: PyTorch-based Hybrid `CNN + LSTM + Transformer` model predicting Focus, Attention, Stress, Fatigue, Cognitive Load, and 5-class Emotion.
- **Explainable AI (XAI)**: Live Feature Attribution (SHAP-proxy) highlighting top frequency bands driving the predictions.
- **Advanced Telemetry**: Brain region heat maps, per-channel quality/noise monitoring, and signal-to-noise ratio (SNR) analytics.
- **Research Mode**: Toggle detailed insights, intermediate layer activations, feature vectors, and advanced system metrics.
- **Experiment Manager**: Save/Load sessions, PDF/CSV export generation, and comprehensive dataset analytics (DEAP, SEED, PhysioNet).
- **Security**: JWT-based Authentication, Role-based Access Control, and SQLite/MongoDB dual-database architecture.

## 🧠 Technology Stack

- **Frontend**: React, TypeScript, Tailwind CSS, Framer Motion, Recharts
- **Backend**: FastAPI, Python 3.10, PyTorch, MNE-Python, Scikit-learn
- **Databases**: MongoDB (Motor Async) for Profiles, SQLite for Time-series Telemetry
- **Deployment**: Docker, Docker Compose, Nginx, GitHub Actions

## ⚙️ Quick Start (Docker)

The easiest way to run the entire stack is using Docker Compose.

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-link.git
cd neural-link

# Start the services (Frontend, Backend, MongoDB)
docker-compose up --build
```
- **Frontend Dashboard**: `http://localhost:80`
- **Backend API Docs**: `http://localhost:8000/docs`

## 🛠️ Manual Local Setup

### 1. Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend
```bash
cd frontend
npm install
npm run dev
```

## 🧪 Testing

The platform includes a comprehensive test suite covering the FastAPI endpoints and the PyTorch ML pipeline.

```bash
cd backend
pytest tests/
```

## 📡 API Overview

- `GET /ws` : WebSocket endpoint streaming continuous 19-channel EEG signals and deep learning predictions at 25Hz.
- `POST /simulate` : Update active simulation dataset (e.g., switch to DEAP/SEED/PhysioNet).
- `POST /train` : Trigger the PyTorch model training loop and save new `.pth` weights.
- `GET /logs` : Retrieve paginated telemetry logs from SQLite.
- `POST /auth/token` : JWT Authentication login.

## 📂 Project Structure
- `/backend`: FastAPI server, ML pipelines, SQLite/MongoDB drivers, PyTorch models.
- `/frontend`: React application, UI components, responsive Grid layouts.
- `/data`: (Auto-created) Directory where EEG datasets like PhysioNet are downloaded.
- `/weights`: (Auto-created) Directory where trained `.pth` model weights are stored.

---
*Built for final-year B.Tech projects, hackathons, and cutting-edge BCI research.*
