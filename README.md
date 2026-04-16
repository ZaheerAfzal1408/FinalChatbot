# 🏗️ THE FOREMAN: Unified Industrial AI

**The Foreman** is a high-performance, consolidated industrial monitoring system. It leverages **LSTM Autoencoders** and a **Multi-Agent AI reasoning chain** to provide 360-degree protection for Oil Refineries, Cold Chain Logistics, and Fire Safety Networks.

---

## 🌟 Key Capabilities

### 🛡️ 1. Refinery Tank Monitoring
- **Real-time Level Tracking**: Analyzes oil levels in Physical01–13.
- **Drift Detection**: Identifies "Impossible Values", "Sudden Spikes", and "Sensor Faults".
- **Dynamic Baselines**: Automatically adjusts thresholds for different oil types or equipment upgrades.

### ❄️ 2. ColdRoom Environment Guard
- **Temp/Humi Stability**: Monitors thermal gradients to prevent product spoilage.
- **Night/Weekend Sensitivity**: Adjusts alert intensity based on facility hours and peak loads.

### ⚠️ 3. Hierarchical Smoke & Fire Detection
- **Network Scanning**: Scans Zones, Rooms, and individual sensor nodes.
- **Multi-Factor Logic**: Combines AI reconstruction errors with hardware warning strings (`mute`, `fault`, `low-vol`) for precise incident reporting.

---

## 🧠 Brain Architecture: The AI Multi-Agent System

The system uses **LangGraph** to orchestrate a specialist team of AI agents:

1. **The Supervisor (Foreman)**: The entry point. It detects user intent and routes queries to the correct domain expert (Tank, ColdRoom, or Smoke).
2. **Domain Experts**: Specialist nodes that execute heavy LSTM inference in the background.
3. **The Investigator (Root Cause)**: Analyzes the raw sensor anomalies to find underlying facility patterns.
4. **The Advisor (Safety)**: Provides actionable, industrial-grade recommendations based on detected incidents.

---

## 🚀 Getting Started

### Backend Setup (Python)
The backend is a unified **FastAPI** service.

1. **Install Dependencies**:
   ```bash
   pip install tensorflow pandas numpy scikit-learn langchain-groq langgraph fastapi uvicorn python-dotenv
   ```
2. **Environment Configuration**: Create a `.env` file:
   ```env
   GROQ_API_KEY=your_key_here
   POSTGRES_URI=your_db_uri
   ```
3. **Launch the Service**:
   ```bash
   cd backend
   python api_chat.py  # Starts the Multi-Agent Chatbot
   python app.py       # Starts the Automated Weekly Pipeline
   ```

### Frontend Setup (React)
The frontend provides a real-time "Command Center" view.
```bash
cd frontend
npm install
npm run dev
```

---

## 📂 Project Structure

- **`backend/core/`**: Shared asset mapping, feature engineering, and status scoring.
- **`backend/specialists/`**: LSTM diagnostic tools for industrial and safety domains.
- **`backend/database/`**: Optimized fetchers for OpenRemote PostgreSQL telemetry.
- **`backend/models/`**: Versioned LSTM Autoencoders (`.h5` and `.pkl` artifacts).

---

## 🔧 Maintenance Commands

- **Force System Realignment**: Use this to reset all AI baselines immediately if operations change.
  ```bash
  python realign_system.py
  ```

---

Developed for **Enterprise Industrial Safety**.
"The Foreman: Always Watching, Always Learning."
