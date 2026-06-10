import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# 1. 自动加载本地环境变量（读取 .env 文件）
load_dotenv()

# 2. 初始化 FastAPI 应用实例
app = FastAPI(
    title="Wear&Care AI Insights API",
    description="微服务接口：将物联网（IoT）传感器的原始数据流转化为管理仪表盘所需的结构化 JSON 指标。",
    version="1.0.0"
)

# 3. 配置跨域资源共享（CORS），让你的前端 Dashboard 网页可以安全地跨域访问此接口
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在真实生产环境中，Wear&Care 会将其严格限制为自己的官网域名
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)


# 4. 定义前端发来的请求体（Request Payload）应该长什么样
class SensorDataPayload(BaseModel):
    raw_logs: str = Field(..., description="来自养老院的物联网传感器原始文本日志。")


# 5. 定义大模型在返回时必须严格遵守的 JSON 结构模型
class CareReport(BaseModel):
    room_number: str = Field(description="养老院的房间号，例如 'Room 302'")
    patient_name: str = Field(description="老人的姓名")
    summary: str = Field(description="给部门经理看的一句话自然语言总结。字数控制在100字以内。")
    anomaly_detected: bool = Field(
        description="如果日志中包含异常行为、异常模式或需要紧急升级的情况，则设为 True；否则为 False。")
    care_suggestion: str = Field(description="给当班护工的下一步具体行动建议或护理指南。")


# 6. 创建让前端 Dashboard 调用的真实 API 路由路径（Endpoint）
@app.post("/api/v1/insights/generate", response_model=CareReport)
async def generate_dashboard_insights(payload: SensorDataPayload):
    """
    接收原始传感器文本数据，通过 OpenAI 的结构化输出机制进行分析，
    并返回一个为管理仪表盘完美优化的干净 JSON 对象。
    """
    # 初始化客户端，OpenAI 的 SDK 会自动寻找并读取环境变量
    client = OpenAI()

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="服务器环境中缺失 OpenAI API Key，请检查 .env 文件。")

    try:
        # 触发大模型的结构化解析功能
        response = client.beta.chat.completions.parse(
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
            response_format=CareReport,  # 【核心】逼迫大模型强制对齐上方定义好的 Pydantic 结构
            temperature=0.0  # 设为 0.0 让结果绝对严谨，防止医疗护理数据出现 AI 幻觉
        )

        # FastAPI 会极其聪明地将解析好的 Pydantic 对象自动序列化为标准的 JSON 字符串返回给客户端！
        return response.choices[0].message.parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 解析时发生内部错误: {str(e)}")


# 7. 基础的服务器健康检查路由
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "WearCare-AI-Layer"}
