# Wear&Care AI Insights Layer (Prototype)

An industrial-grade AI microservice built with **FastAPI** and **OpenAI Structured Outputs** to transform raw IoT elder-care sensor telemetry into structured, deterministic, and actionable natural-language insights for management dashboards.

---

## 🚀 Business Value & Problem Solved
In modern elder-care facilities, IoT sensors generate massive streams of raw, unstructured data (e.g., incontinence alerts, mobility metrics). Care home department managers do not have the time to audit raw logs manually. 

This microservice acts as the intelligent bridge:
1. **Intercepts Raw Telemetry**: Accepts raw unstructured event text streams.
2. **Enforces Zero-Hallucination Schemas**: Utilizes OpenAI's latest Structured Outputs to force the LLM to strictly populate a structured data model without hallucinating clinical details.
3. **Empowers Caregivers**: Outputs clean JSON payloads containing clinical summaries and actionable care suggestions, enabling the frontend dashboard to render dynamic charts and instant red-flag alerts.

---

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Web Framework:** FastAPI (Asynchronous API endpoints)
* **AI Layer:** OpenAI API (`gpt-4o-mini` with Structured Outputs via Beta Parse)
* **Data Validation:** Pydantic v2 (Strict schema and type enforcement)
* **Server:** Uvicorn (ASGI web server)
* **Configuration:** Python-Dotenv (Decoupled environment variables)

---

## 📁 Project Architecture
```text
Wear_Care/
│
├── .env                  # Local secrets (API Keys) - NEVER uploaded to GitHub
├── .gitignore            # Git exclusion rules (protects .env from leaks)
├── requirements.txt      # Project library dependencies
├── main.py               # common API
├── server.py             # Live FastAPI API Server with Pydantic schemas
└── README.md             # Project documentation (You are here)
```

---

## ⚡ Quick Start & Installation

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com
cd Wear_Care
python -m venv .venv
```

Activate the virtual environment:
* **Windows (CMD):** `.venv\Scripts\activate`
* **macOS/Linux:** `source .venv/bin/activate`

### 2. Install Dependencies
```bash
pip install fastapi uvicorn openai pydantic python-dotenv
```

### 3. Configure Environment Variables
Create a file named `.env` in the root directory and add your secret token:
```env
OPENAI_API_KEY=sk-proj-your_actual_openai_api_key_here
```

### 4. Fire Up the Server
Launch the live Uvicorn microservice:
```bash
uvicorn server:app --reload
```

---

## 🧪 Interactive API Documentation & Testing

Once the server is running, FastAPI automatically compiles interactive Swagger UI documentation. 

Navigate to: **`http://127.0.0`**

### Sample Endpoint: `POST /api/v1/insights/generate`

**Expected Request Payload (JSON):**
```json
{
  "raw_logs": "[TIMESTAMP: 2026-06-09]\nDEVICE_ID: INCONT-SENSOR-99\nLOCATION: Room 302 (Resident: Anna Jensen)\nEVENT_LOG:\n- 02:15 : Incontinence sensor triggered (Level: High)\n- 02:20 : Roll-over frequency high (3 times in 5 mins)\n- 03:40 : Incontinence sensor triggered (Level: High)\nHISTORICAL_AVG_DAILY_TRIGGERS: 1.5"
}
```

**Deterministic Server Response (Code 200 JSON):**
```json
{
  "room_number": "Room 302",
  "patient_name": "Anna Jensen",
  "summary": "Anna Jensen experienced multiple high-level incontinence sensor triggers and an increased roll-over frequency, indicating potential discomfort or need for attention.",
  "anomaly_detected": true,
  "care_suggestion": "Monitor Anna closely for further signs of discomfort and consider adjusting her care plan to address her needs."
}
```

---

## 🔒 Security & Compliance Compliance
* **Data Privacy:** Hardcoded credentials are strictly banned. All tokens live outside the codebase via `.env` files.
* **Input Interception:** FastAPI automatically catches structural mismatch anomalies and blocks malformed JSON inputs instantly, returning a native `422 Validation Error` to safeguard the LLM infrastructure from unnecessary execution costs.
