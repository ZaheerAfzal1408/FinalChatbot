# The Foreman: Hierarchical Multi-Agent Industrial Monitoring System

**The Foreman** is a high-fidelity industrial anomaly detection platform designed for refinery tanks and coldroom management. It utilizes a hierarchical multi-agent architecture to bridge the gap between raw sensor data and human-readable industrial diagnostics.

---

## 🚀 Key Features

*   **Autonomous Oversight**: A background pipeline that fetches data, trains models, and predicts anomalies every 3 hours.
*   **Deep-Learning Engine**: Uses **LSTM Autoencoders** to detect subtle behavioral shifts (not just simple spikes).
*   **Hierarchical Agents**: Managed by **LangGraph**, the system coordinates between specialized agents for different asset types.
*   **Instant Diagnostics**: Real-time chat interface that provides root-cause explanations and technical insights (MSE, Threshold, ROC).
*   **Batch Inference**: Analyzes entire windows of data to identify historical trends and persistent leaks.

---

## 🛠 Technology Stack

### Backend & AI
- **FastAPI**: High-performance API orchestration.
- **LangGraph & LangChain**: Multi-agent framework for intelligent task routing.
- **TensorFlow/Keras**: Deep learning for time-series anomaly detection.
- **PostgreSQL**: Primary data store for IoT sensor telemetry.
- **Scikit-Learn & Joblib**: Data scaling and artifact management.

### Frontend
- **React 18**: Dynamic, glassmorphic UI.
- **Vite**: Ultra-fast build tool.
- **Framer Motion**: Smooth micro-animations for a premium feel.
- **Vanilla CSS**: Bespoke styling for maximum performance.

---

## 📂 Project Structure

```text
├── backend/            # Primary ML Pipeline & Data Hub
│   ├── data/           # Raw sensor telemetry (CSV/DB)
│   ├── models/         # Persistent LSTM models & Scalers
│   ├── database/       # DB connection & fetch utilities
│   └── app.py          # The Autonomous Overseer (Scheduler)
├── chatbot/            # Multi-Agent API
│   ├── api.py          # FastAPI server for the chatbot
│   └── tools_industrial.py # Logic Bridge between Chat and Backend
└── frontend/           # React Application
    └── src/components/ # Bespoke UI components (Chat, Sidebar, etc.)
```

---

## 🏁 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ and Node.js installed.

### 2. Environment Setup
Create a `.env` file in the `backend/` and `chatbot/` directories:
```env
POSTGRES_URI=your_database_connection_string
GROQ_API_KEY=your_groq_api_key
```

### 3. Installation
**Backend/Chatbot:**
```bash
cd chatbot
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

### 4. Running the System
You need to run three components simultaneously:
1. **Background Pipeline**: `python backend/app.py`
2. **Chatbot API**: `python chatbot/api.py`
3. **Frontend Dashboard**: `npm run dev`

---

## 🧠 Core Logic: LSTM Autoencoder
The system uses an **Autoencoder** architecture which:
1.  **Encodes** a window of "Normal" behavior into a compressed representation.
2.  **Decodes** it back into original sensor values.
3.  **Calculates Error (MSE)**: If the error between the input and reconstructed output is higher than a dynamically calculated **Threshold**, an anomaly is triggered.

---

## 📜 License
Internal Industrial Use Only. Designed by **The Foreman Team**.
