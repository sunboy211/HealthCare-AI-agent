import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI



app = FastAPI(
    title="Wear&Care AI Insights API",
    description="Microservice Interface: Transforms raw data streams from Internet of Things (IOT) sensors into structured JSON metrics required for management dashboards.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://wearcare.com"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["POST", "GET"],
)

class SensorDataPayload(BaseModel):
    raw_logs: str = Field(..., description="Raw text logs from IOT sensors in nursing homes.")


class CareReport(BaseModel):
    room_number: str = Field(description="The room number of the nursing home, for example, 'Room 302' ")
    patient_name: str = Field(description="The name")
    summary: str = Field(description="A one-sentence summary in natural language for the department manager. Keep it under 100 words.")
    anomaly_detected: bool = Field(
        description="Set to True if the logs contain abnormal behavior, abnormal patterns, or situations requiring urgent escalation; otherwise, set to False.")
    care_suggestion: str = Field(description="Specific next steps or nursing guidelines for the on-duty caregiver.")


@app.post("/api/v1/insights/generate", response_model=CareReport)
async def generate_dashboard_insights(payload: SensorDataPayload):
    """
    Receive raw IoT logs for large-scale model analysis.
    If an anomaly is detected (anomaly_detected == True), immediately bombard all front-end dashboards via WebSocket.
    """
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )

    #if not os.getenv("OPENAI_API_KEY"):
        #raise HTTPException(status_code=500, detail="The OpenAI API Key is missing from your server environment. Please check your .env file.")

    try:
        response = client.beta.chat.completions.parse(
            model="phi3",
            messages=[{ "role": "system",
            "content": "You are an expert AI clinical assistant specialized in processing IoT welfare-tech sensor data. Parse the raw logs and strictly populate the requested structured report in English."},
            {"role": "user", "content": payload.raw_logs}],
            response_format=CareReport,
            temperature=0.0
        )

        return response.choices[0].message.parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred during LLM parsing: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "WearCare-AI-Layer"}
