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

import core.asset_mapping as am
from specialists.tools_industrial import analyze_coldroom, analyze_tank, scan_all_coldrooms, scan_all_tanks
from specialists.tools_smoke import analyze_smoke_incident, scan_all_smoke_alarms

# Environment Setup
load_dotenv()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

# Native imports from unified backend
try:
    from app import slugify
except ImportError as e:
    logging.error(f"Error importing from app.py: {e}")
    def slugify(text):
        import re
        return re.sub(r'[^a-zA-Z0-9]', '', str(text)).lower()

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
    root_cause_analysis: str
    recommendations: str
    final_response: str
    check_all: bool
    force_retrain: bool

def supervisor_node(state: AgentState):
    """The Root Supervisor that routes to experts with guardrails."""
    last_message = state['messages'][-1].content
    query_lower = last_message.lower()
    
    force_retrain = any(word in query_lower for word in ["retrain", "reset", "update baseline", "re-train"])
    
    if any(phrase in query_lower for phrase in ["smoke", "fire", "canteen", "zone"]):
        logger.info("Foreman: Smoke/Fire safety intent detected.")
        choice = 'smoke_expert'
        return {"next_node": choice, "check_all": "all" in query_lower, "force_retrain": force_retrain}

    if any(phrase in query_lower for phrase in ["any last incident", "all incidents", "scan all", "status of all"]):
        logger.info(f"Guardrail trigger: Global scan requested.")
        choice = 'coldroom_expert' if 'coldroom' in query_lower else 'tank_expert'
        return {"next_node": choice, "check_all": True, "force_retrain": force_retrain}

    if not am.ASSET_CACHE: am.load_dynamic_mappings()
    for asset_name in am.ASSET_CACHE.values():
        if asset_name.lower() in query_lower:
            logger.info(f"Guardrail trigger: '{asset_name}' detected. Routing to expert.")
            choice = 'coldroom_expert' if 'coldroom' in asset_name.lower() else 'tank_expert'
            return {"next_node": choice, "check_all": False, "force_retrain": force_retrain}

    next_node = 'FINISH' # Default for general greetings or non-industrial queries
    prompt = f"""Identify the user's intent:
    - If they ask about coldrooms, set next_node to 'coldroom_expert'.
    - If they ask about tanks/refinery, set next_node to 'tank_expert'.
    - If they ask about smoke, fire, or safety sensors, set next_node to 'smoke_expert'.
    - If it's a general greeting, set next_node to 'FINISH'.
    
    Return ONLY 'coldroom_expert', 'tank_expert', 'smoke_expert', or 'FINISH'.
    Identify intent for: {last_message}"""
    response = llm.invoke(prompt).content.strip().lower()
    if 'coldroom' in response: next_node = 'coldroom_expert'
    elif 'tank' in response: next_node = 'tank_expert'
    elif 'smoke' in response: next_node = 'smoke_expert'
        
    return {"next_node": next_node, "check_all": False, "force_retrain": force_retrain}

def coldroom_expert_node(state: AgentState):
    """Expert node for ColdRoom diagnostics."""
    query = state['messages'][-1].content
    matches = re.findall(r'cold\s*room\s*(\d*)', query.lower())
    outputs = []
    if state.get("check_all", False) or not matches:
        outputs.append(scan_all_coldrooms(fetch_hours=3))
    else:
        for num in matches:
            num = num.strip() if num.strip() else "1"
            alias = f"coldroom{num}"
            aid = am.get_asset_id(alias)
            asset_name = am.get_asset_name(aid) if aid else alias
            report = analyze_coldroom(asset_name, force_retrain=state.get("force_retrain", False), fetch_hours=3)
            outputs.append(report)
    return {"tool_outputs": outputs, "next_node": "root_cause"}

def smoke_expert_node(state: AgentState):
    """ Expert node for Smoke & Fire safety diagnostics. """
    query = state['messages'][-1].content.lower()
    outputs = []
    
    # 1. Detect rooms (e.g., "canteen", "security room")
    found_assets = []
    if not am.SMOKE_MAPPINGS: am.load_dynamic_mappings()
    for aid, name in am.SMOKE_MAPPINGS.items():
        if name.lower() in query:
            found_assets.append(name)
            
    if state.get("check_all", False) or not found_assets:
        outputs.append(scan_all_smoke_alarms(fetch_hours=3))
    else:
        for name in found_assets:
            report = analyze_smoke_incident(name, force_retrain=state.get("force_retrain", False), fetch_hours=3)
            outputs.append(report)
            
    return {"tool_outputs": outputs, "next_node": "root_cause"}

def tank_expert_node(state: AgentState):
    """Expert node for Refinery Tank diagnostics."""
    query = state['messages'][-1].content
    matches = re.findall(r'(?:tank|physical|refinery)\s*(\d*)', query.lower())
    outputs = []
    if state.get("check_all", False) or not matches:
        outputs.append(scan_all_tanks(fetch_hours=3))
    else:
        for num in matches:
            num = num.strip() if num.strip() else "1"
            alias = f"tank{num}"
            aid = am.get_asset_id(alias)
            asset_name = am.get_asset_name(aid) if aid else alias
            report = analyze_tank(asset_name, force_retrain=state.get("force_retrain", False), fetch_hours=3)
            outputs.append(report)
    return {"tool_outputs": outputs, "next_node": "root_cause"}

def root_cause_node(state: AgentState):
    """Reasoning node to identify patterns and root causes (The Investigator)."""
    outputs = state.get("tool_outputs", [])
    valid_sensors = []
    for o in outputs:
        if isinstance(o, dict):
            if "all_reports" in o: valid_sensors.extend(o["all_reports"])
            else: valid_sensors.append(o)
    if not valid_sensors: return {"root_cause_analysis": "No diagnostic data."}
    prompt = f"Analyze root causes: {valid_sensors}. Be concise."
    analysis = llm.invoke(prompt).content
    return {"root_cause_analysis": analysis}

def recommend_node(state: AgentState):
    """Advisory node for actionable industrial recommendations (The Safety Advisor)."""
    root_cause = state.get("root_cause_analysis", "")
    prompt = f"Recommendations for: {root_cause}. 3-4 points."
    recs = llm.invoke(prompt).content
    return {"recommendations": recs}

# --- Graph Construction ---
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("coldroom_expert", coldroom_expert_node)
workflow.add_node("tank_expert", tank_expert_node)
workflow.add_node("smoke_expert", smoke_expert_node)
workflow.add_node("root_cause", root_cause_node)
workflow.add_node("recommend", recommend_node)
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", lambda x: x["next_node"], {
    "coldroom_expert": "coldroom_expert", 
    "tank_expert": "tank_expert", 
    "smoke_expert": "smoke_expert",
    "FINISH": END
})
workflow.add_edge("coldroom_expert", "root_cause")
workflow.add_edge("tank_expert", "root_cause")
workflow.add_edge("smoke_expert", "root_cause")
workflow.add_edge("root_cause", "recommend")
workflow.add_edge("recommend", END)
graph = workflow.compile()

@app.post("/api/chat")
async def chat_handler(request: ChatRequest):
    """Unified Entry Point for the Multi-Agent System."""
    logger.info(f"Industrial Data Pipeline triggered: '{request.message}'")
    try:
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
            "next_node": "",
            "tool_outputs": [],
            "root_cause_analysis": "",
            "recommendations": "",
            "final_response": "",
            "check_all": False,
            "force_retrain": False
        }
        
        # Execute the Graph
        result = graph.invoke(initial_state)
        tool_outputs = result.get("tool_outputs", [])
        
        # Synthesize final response using Groq
        if tool_outputs:
            # Flatten outputs for incident detection
            flattened = []
            for o in tool_outputs:
                if isinstance(o, dict) and "all_reports" in o: flattened.extend(o["all_reports"])
                elif isinstance(o, list): flattened.extend(o)
                else: flattened.append(o)

            is_incident = any(
                isinstance(o, dict) and o.get("status") in ["Anomaly", "Critical", "Warning", "incident"]
                for o in flattened
            )
            
            show_tech = is_incident or any(x in request.message.lower() for x in ["incident reading", "mse", "threshold", "technical"])
            
            root_cause = result.get("root_cause_analysis", "")
            recommendations = result.get("recommendations", "")
            
            summary_prompt = f"""
            You are 'The Foreman', an Industrial Multi-Agent Supervisor. 
            Original Query: "{request.message}"
            
            INDUSTRIAL DIAGNOSTICS:
            - Tool Data: {tool_outputs}
            - Root Cause Analysis (from Investigator): {root_cause}
            - Safety Recommendations (from Advisor): {recommendations}
            
            DIAGNOSTIC RULES:
            1. **DATA VISIBILITY**: Include the sensor readings bullet points.
            2. **STRUCTURE**: Use bold headers: ### Diagnostic Summary, ### Root Cause Analysis, ### Foreman's Recommendations.
            3. **STATUS**: Use 🔴 **CRITICAL**, 🟡 **WARNING**, 🟢 **OPERATIONAL**, or ⚪ **NO DATA AVAILABLE**.
            4. **TECHNICAL**: {'Include the MSE and Threshold clearly.' if show_tech else 'Hide technical MSE values.'}
            5. Integrate the investigator's findings and the advisor's actions seamlessly.
            """
            final_response = llm.invoke(summary_prompt).content
        else:
            # Check if an agent gave a specific reason for no results (e.g. room not found)
            last_msg = result['messages'][-1]
            if isinstance(last_msg, AIMessage) and last_msg.content != request.message:
                final_response = last_msg.content
            else:
                final_response = "I am 'The Foreman'. I can help you monitor Smoke Alarms. Which room should I check for you?"
            
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
