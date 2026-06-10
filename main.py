import os
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
# 1. Define the structured JSON response layout that the management dashboard expects
class CareReport(BaseModel):
    room_number: str = Field(description="The care home room number, e.g., 'Room 302'")
    patient_name: str = Field(description="The name of the resident/patient")
    summary: str = Field(
        description="A concise, 1-sentence natural language summary for the department manager. Max 100 words.")
    anomaly_detected: bool = Field(
        description="Set to True if the log indicates unusual behavior, patterns, or critical escalation. Otherwise False.")
    care_suggestion: str = Field(
        description="Actionable next-step recommendation or instruction for the caregivers on duty.")


def generate_json_report(sensor_data: str):
    # Ensure you replace this with your actual API key
    client = OpenAI()

    try:
        # 2. Utilize OpenAI's Structured Outputs via the Beta parse API
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert AI clinical assistant specialized in processing IoT welfare-tech sensor data. Parse the raw logs and strictly populate the requested structured report in English."
                },
                {
                    "role": "user",
                    "content": sensor_data
                }
            ],
            response_format=CareReport,  # Forces the model to strictly match the Pydantic schema
            temperature=0.0  # Deterministic output to prevent hallucination in care data
        )

        # 3. Extract and return the parsed Python object directly
        return response.choices[0].message.parsed

    except Exception as e:
        return f"Error encountered during API parsing: {e}"


if __name__ == "__main__":
    # Simulating a raw Wear&Care IoT sensor alert stream
    mock_sensor_stream = """
    [TIMESTAMP: 2026-06-09]
    DEVICE_ID: INCONT-SENSOR-99
    LOCATION: Room 302 (Resident: Anna Jensen)
    EVENT_LOG:
    - 02:15 : Incontinence sensor triggered (Level: High)
    - 02:20 : Roll-over frequency high (3 times in 5 mins)
    - 03:40 : Incontinence sensor triggered (Level: High)
    - 05:10 : Incontinence sensor triggered (Level: Medium)
    HISTORICAL_AVG_DAILY_TRIGGERS: 1.5
    """

    print("Converting raw IoT data stream into dashboard-compatible structured JSON...")
    report = generate_json_report(mock_sensor_stream)

    # Printing individual fields (How backend routes the variables)
    print("\n--- Successfully Extracted Fields ---")
    print(f"Room Number: {report.room_number}")
    print(f"Patient Name: {report.patient_name}")
    print(f"Manager Summary: {report.summary}")
    print(f"Anomaly Flagged: {report.anomaly_detected}")
    print(f"Caregiver Action: {report.care_suggestion}")

    # Printing the exact JSON payload payload that gets sent to the Frontend React Dashboard
    print("\n--- Raw JSON Payload for Frontend Integration ---")
    print(report.model_dump_json(indent=2))
