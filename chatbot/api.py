import os
import sys
import logging
import re
from typing import List, Optional, TypedDict, Annotated
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from groq import Groq
from dotenv import load_dotenv

# Import LangGraph components
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Import mappings and specialist tools from root
import asset_mapping as am
from tools_industrial import analyze_coldroom, analyze_tank

# Environment Setup
load_dotenv()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
# Robust path resolution for backend components
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
backend_path = os.path.join(project_root, 'backend')
if backend_path not in sys.path:
    sys.path.append(backend_path)

# Import from backend for ML logic
try:
    from app import slugify
except ImportError as e:
    logging.error(f"Error importing from backend: {e}")

# Global LLM Configuration
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TIME_STEPS = 30

app = FastAPI(title="The Foreman: Multi-Agent Industrial Supervisor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: list = []

# --- 🧠 LANGGRAPH MULTI-AGENT STATE MACHINE ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The chat history"]
    next_node: str
    tool_outputs: List[dict]
    final_response: str

def supervisor_node(state: AgentState):
    """The Root Supervisor that routes to experts."""
    last_message = state['messages'][-1].content
    
    prompt = f"""
    You are 'The Foreman', an Industrial Multi-Agent Supervisor.
    User Query: {last_message}
    
    Identify the user's intent:
    - If they ask about coldrooms (e.g., 'Check coldroom 1'), set next_node to 'coldroom_expert'.
    - If they ask about tanks/refinery (e.g., 'Status of Tank 1'), set next_node to 'tank_expert'.
    - If user asks about anything about previous query, then analyze that query and provide the response.
    - If it's a general greeting, set next_node to 'FINISH'.
    
    Return ONLY the node name: 'coldroom_expert', 'tank_expert', or 'FINISH'.
    """
    
    response = llm.invoke(prompt).content.strip().lower()
    
    if 'coldroom' in response:
        next_node = 'coldroom_expert'
    elif 'tank' in response:
        next_node = 'tank_expert'
    else:
        next_node = 'FINISH'
        
    return {"next_node": next_node}

def coldroom_expert_node(state: AgentState):
    """Expert node for ColdRoom diagnostics."""
    query = state['messages'][-1].content
    # Find all coldrooms mentioned (handles "cold room 1", "coldroom 1", "coldroom")
    matches = re.findall(r'cold\s*room\s*(\d*)', query.lower())
    
    outputs = []
    for num in matches:
        # Default to "2" if no number is given, but "cold room" was mentioned
        num = num.strip() if num.strip() else "2"
        asset_name = f"coldroom{num}"
        logger.info(f"Triggering Pipeline for {asset_name}...")
        report = analyze_coldroom(asset_name)
        outputs.append(report)
        
    return {"tool_outputs": outputs, "next_node": "FINISH"}

def tank_expert_node(state: AgentState):
    """Expert node for Refinery Tank diagnostics."""
    query = state['messages'][-1].content
    # Find all tanks/refinery mentioned (handles "tank 1", "refinery 1", etc.)
    matches = re.findall(r'(?:tank|physical|refinery)\s*(\d*)', query.lower())
    
    outputs = []
    for num in matches:
        # Default to "1" if no number is given
        num = num.strip() if num.strip() else "1"
        asset_name = f"tank{num}"
        logger.info(f"Triggering Pipeline for {asset_name}...")
        report = analyze_tank(asset_name)
        outputs.append(report)
        
    return {"tool_outputs": outputs, "next_node": "FINISH"}

# --- Graph Construction ---
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("coldroom_expert", coldroom_expert_node)
workflow.add_node("tank_expert", tank_expert_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next_node"],
    {
        "coldroom_expert": "coldroom_expert",
        "tank_expert": "tank_expert",
        "FINISH": END
    }
)

workflow.add_edge("coldroom_expert", END)
workflow.add_edge("tank_expert", END)

graph = workflow.compile()

# --- API ENDPOINTS ---

@app.post("/api/chat")
async def chat_handler(request: ChatRequest):
    """Unified Entry Point for the Multi-Agent System."""
    logger.info(f"Industrial Data Pipeline triggered: '{request.message}'")
    try:
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
            "next_node": "",
            "tool_outputs": [],
            "final_response": ""
        }
        
        # Execute the Graph
        result = graph.invoke(initial_state)
        tool_outputs = result.get("tool_outputs", [])
        
        # Synthesize final response using Groq
        if tool_outputs:
            is_anomaly = any(
                isinstance(o, dict) and o.get("status") == "Anomaly" 
                for o in tool_outputs
            )
            show_tech = is_anomaly or any(x in request.message.lower() for x in ["anomaly reading", "mse", "threshold", "technical"])
            
            summary_prompt = f"""
            You are 'The Foreman', an Industrial Multi-Agent Supervisor. 
            The user's original query was: "{request.message}"
            
            Based on these industrial tool outputs: {tool_outputs}, provide a professional diagnostic response.
            
            DIAGNOSTIC RULES:
            1. **DATA VISIBILITY**: Always include the sensor readings (Temperature, Level, etc.) in bullet points.
            2. **ANOMALY EXPLANATION**: If an anomaly is detected, focus on explaining **WHEN** it was observed and **WHY** (based on the MSE and Threshold).
            3. **TECHNICAL Bullets**: {'Include the MSE and Threshold values clearly in a "Technical Diagnostic" section.' if show_tech else 'Keep the report professional and hide technical MSE/Threshold values unless specifically requested.'}
            4. Use status indicators: 🔴 **ANOMALY DETECTED** or 🟢 **OPERATIONAL**.
            5. Use bold headers (### Diagnostic Summary, ### Technical Diagnostic, etc.).
            6. Provide a 'Foreman's Recommendation' at the end.
            7. Tone: Professional, Industrial, Objective.
            """
            final_response = llm.invoke(summary_prompt).content
        else:
            final_response = "I am 'The Foreman'. I can help you monitor ColdRooms and Refinery Tanks. Which asset should I check for you?"
            
        return {
            "reply": final_response,
            "data": tool_outputs[0] if len(tool_outputs) == 1 else {"multi_report": tool_outputs}
        }
    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
