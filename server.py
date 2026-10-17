import os
import logging
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, OpenAIError
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WearCareAPI")
load_dotenv()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Nurse online. Online device number: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Nurse offline. Online device number: {len(self.active_connections)}")

    async def broadcast_alert(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to push data to a frontend: {str(e)}")

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("Critical: OPENAI_API_KEY is missing!")
        raise RuntimeWarning("Missing OPENAI_API_KEY")

    app.state.openai_client = AsyncOpenAI()
    logger.info("OpenAI Asynchronous client initialization successful.")
    yield
    await app.state.openai_client.close()

app = FastAPI(
    title="Wear&Care AI Insights API with Real-time Alerts",
    description="Microservice Interface: Transforms raw data streams from Internet of Things (IOT) sensors into structured JSON metrics required for management dashboards.",
    version="1.0.0"
)

# Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://wearcare.com"],  # Wear&Care url
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["POST", "GET"],
)


# Define what the request payload sent from the frontend
class SensorDataPayload(BaseModel):
    raw_logs: str = Field(..., description="Raw text logs from IOT sensors in nursing homes.")


# Define the JSON structure model that the large model must strictly adhere to when returning it.
class CareReport(BaseModel):
    room_number: str = Field(description="The room number of the nursing home, for example, 'Room 302' ")
    patient_name: str = Field(description="The name")
    summary: str = Field(description="A one-sentence summary in natural language for the department manager. Keep it under 100 words.")
    anomaly_detected: bool = Field(
        description="Set to True if the logs contain abnormal behavior, abnormal patterns, or situations requiring urgent escalation; otherwise, set to False.")
    care_suggestion: str = Field(description="Specific next steps or nursing guidelines for the on-duty caregiver.")


# Create the actual API route path (Endpoint) that the front-end Dashboard calls.
@app.post("/api/v1/insights/generate", response_model=CareReport)
async def generate_dashboard_insights(payload: SensorDataPayload):
    """
    Receive raw IoT logs for large-scale model analysis.
    If an anomaly is detected (anomaly_detected == True), immediately bombard all front-end dashboards via WebSocket.
    """
    client: AsyncOpenAI = app.state.openai_client

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="The OpenAI API Key is missing from your server environment. Please check your .env file.")

    try:
        response = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert AI clinical assistant specialized in processing IoT welfare-tech sensor data. Parse the raw logs and strictly populate the requested structured report in English."
                },
                {
                    "role": "user",
                    "content": payload.raw_logs
                }
            ],
            response_format=CareReport,  # Predefined Pydantic structure
            temperature=0.0,
            timeout = 30.0
        )

        # FastAPI intelligent in serializing the parsed Pydantic object into a standard JSON string and returning it to the client
        report: CareReport = response.choices[0].message.parsed
        if report.anomaly_detected:
            logger.warning(f"abnormal detected！Room: {report.room_number}, Elderly: {report.patient_name}")
            alert_payload = {
                "event": "EMERGENCY_ALERT",
                "room_number": report.room_number,
                "patient_name": report.patient_name,
                "summary": report.summary,
                "care_suggestion": report.care_suggestion
            }
            await manager.broadcast_alert(alert_payload)

        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred during LLM parsing: {str(e)}")
    except Exception as e:
        logger.error(f"Internal unknown error: {str(e)}")
        raise HTTPException(status_code=502, detail="Internal server error")


# WebSocket real-time alert
@app.websocket("/ws/alerts")
async def websocket_alerts_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # wait to receive data
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong") # an example
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket Abnormal disconnection: {str(e)}")
        manager.disconnect(websocket)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "WearCare-AI-Layer"}
