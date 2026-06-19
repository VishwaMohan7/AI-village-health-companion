import logging
from typing import TypedDict, Annotated, Sequence, Literal

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from backend.agent.tools import triage_tool, clinic_locator_tool, sms_alert_tool, outbreak_checker_tool
from backend.services.ollama_service import (
    generate_response,
    build_triage_prompt_question,
    build_triage_prompt_summary,
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    patient_id: str
    patient_name: str
    patient_age: str
    patient_language: str
    patient_location: str
    symptoms: str
    urgency_level: str
    actions_taken: list[str]
    next_node: str
    # Optional fields for richer triage conversation
    symptoms_original: str | None
    questions_asked_count: int | None
    triage_answers_summary: str | None

tools = [triage_tool, clinic_locator_tool, sms_alert_tool, outbreak_checker_tool]

def intake_node(state: AgentState):
    """Collects basic patient info."""
    logging.info("--- INTAKE NODE ---")
    
    # In a real flow, this would parse state.messages[-1] to extract details
    # For now, we mock the extraction process for prototype
    
    messages = state.get("messages", [])
    default_symptoms = messages[-1].content if messages else ""
    
    return {
        "patient_name": state.get("patient_name", "Unknown"),
        "patient_age": state.get("patient_age", "Unknown"),
        "patient_language": state.get("patient_language", "en"),
        "symptoms": state.get("symptoms", default_symptoms),
        "symptoms_original": state.get("symptoms_original", state.get("symptoms", default_symptoms)),
        "questions_asked_count": state.get("questions_asked_count", 0),
        "triage_answers_summary": state.get("triage_answers_summary", ""),
        "next_node": "symptom_node"
    }

def symptom_node(state: AgentState):
    """Asks follow-up questions if needed."""
    logging.info("--- SYMPTOM NODE ---")
    
    current_symptoms = state.get("symptoms", "")
    
    # Simple explicit mock for prototype follow-up
    # Real implementation uses llm to decide if more context is needed
    if len(current_symptoms) < 10:
        return {"messages": [AIMessage(content="Could you please tell me more about your symptoms?")], "next_node": END}
        
    return {"next_node": "reasoning_node"}

def reasoning_node(state: AgentState):
    """Runs rule-based triage and prepares for final LLM summary."""
    logging.info("--- REASONING NODE ---")
    
    # Use existing rule-based triage_tool for urgency (keeps original logic)
    try:
        triage_result = triage_tool.invoke({"symptoms": state.get("symptoms", ""), "patient_history": ""})
    except Exception as e:
        logging.error(f"triage_tool failed: {e}")
        triage_result = "URGENCY: MEDIUM. Rationale: fallback."

    urgency = "MEDIUM"
    if "URGENCY:" in triage_result:
        urgency = triage_result.split("URGENCY:")[1].split(".")[0].strip().upper() or "MEDIUM"

    msg = AIMessage(content=triage_result)
    return {
        "urgency_level": urgency,
        "messages": [msg],
        "next_node": "response_node",
    }

def action_node(state: AgentState):
    """Executes tools requested by reasoning node."""
    logging.info("--- ACTION NODE ---")
    
    last_message = state['messages'][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        # Map tool names to actual functions
        tool_map = {t.name: t for t in tools}
        
        tool_responses = []
        actions = state.get("actions_taken", [])
        
        for tool_call in last_message.tool_calls:
            tool_fn = tool_map.get(tool_call['name'])
            if tool_fn:
                try:
                    result = tool_fn.invoke(tool_call['args'])
                    tool_responses.append(
                        ToolMessage(content=str(result), tool_call_id=tool_call['id'], name=tool_call['name'])
                    )
                    actions.append(f"Used {tool_call['name']}: {str(result)[:50]}...")
                    
                    # Update urgency directly from triage_tool response
                    if tool_call['name'] == 'triage_tool' and "URGENCY:" in str(result):
                        extracted_urgency = str(result).split("URGENCY:")[1].split(".")[0].strip()
                        state['urgency_level'] = extracted_urgency
                
                except Exception as e:
                    logging.error(f"Error executing tool {tool_call['name']}: {e}")
                    tool_responses.append(
                        ToolMessage(content=f"Error: {str(e)}", tool_call_id=tool_call['id'], name=tool_call['name'])
                    )
        
        # Return tool responses and go directly to response node
        return {
            "messages": tool_responses, 
            "actions_taken": actions,
            "next_node": "response_node"
        }
        
    return {"next_node": "response_node"}

def response_node(state: AgentState):
    """Formulates final response using local Ollama LLM."""
    logging.info("--- RESPONSE NODE ---")
    
    symptoms = state.get("symptoms", "")
    symptoms_original = state.get("symptoms_original", symptoms)
    urgency = state.get("urgency_level", "MEDIUM")
    actions = state.get("actions_taken", [])
    user_lang = state.get("patient_language", "en-IN")
    q_count = state.get("questions_asked_count", 0)
    answers_summary = state.get("triage_answers_summary", "")
    MAX_QUESTIONS = 3

    # Latest user message in original language if available
    last_user_msg = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break
    if not last_user_msg:
        last_user_msg = symptoms_original

    # Update answers summary with the latest user message (simple concatenation)
    if last_user_msg:
        sep = "\n- "
        if not answers_summary:
            answers_summary = f"- {last_user_msg}"
        else:
            answers_summary = f"{answers_summary}{sep}{last_user_msg}"

    try:
        if (q_count or 0) < MAX_QUESTIONS:
            # QUESTION PHASE: ask exactly one follow-up question
            prompt = build_triage_prompt_question(
                user_message=last_user_msg,
                language_code=user_lang,
                symptoms_summary=symptoms,
                questions_asked_count=q_count,
            )
        else:
            # SUMMARY PHASE: provide final triage summary (no more questions)
            prompt = build_triage_prompt_summary(
                user_message=last_user_msg,
                language_code=user_lang,
                symptoms_summary=symptoms,
                answers_summary=answers_summary,
            )
        text = generate_response(prompt)
        final_answer = AIMessage(content=text)
    except Exception as e:
        logging.error(f"Ollama response failed, falling back to simple text: {e}")
        fallback = (
            f"I have recorded your symptoms ({symptoms}) and assessed your condition as "
            f"{urgency} urgency. Please contact a healthcare provider if you worsen."
        )
        final_answer = AIMessage(content=fallback)

    # Increment question count for next turn only while in QUESTION PHASE
    next_q_count = (q_count or 0) + 1 if (q_count or 0) < MAX_QUESTIONS else q_count
    return {
        "messages": [final_answer],
        "questions_asked_count": next_q_count,
        "triage_answers_summary": answers_summary,
        "next_node": "followup_node",
    }

def followup_node(state: AgentState):
    """Schedules follow-up."""
    logging.info("--- FOLLOWUP NODE ---")
    actions = state.get("actions_taken", [])
    actions.append("Scheduled 24-hour follow-up check-in.")
    return {"actions_taken": actions, "next_node": END}

# Conditional routing
def route_after_intake(state: AgentState) -> Literal["symptom_node"]:
    return state.get("next_node", "symptom_node")
    
def route_after_symptom(state: AgentState) -> Literal["reasoning_node", "__end__"]:
    # Skip immediately to action/action rules if EMERGENCY - handled inside reasoning for now
    if state.get("next_node") == END:
        return END
    return "reasoning_node"

def route_after_reasoning(state: AgentState) -> Literal["action_node", "response_node"]:
    return state.get("next_node", "response_node")

def route_after_action(state: AgentState) -> Literal["reasoning_node", "response_node"]:
    return state.get("next_node", "reasoning_node")

def route_after_response(state: AgentState) -> Literal["followup_node"]:
    return state.get("next_node", "followup_node")

# Build graph
workflow = StateGraph(AgentState)

workflow.add_node("intake_node", intake_node)
workflow.add_node("symptom_node", symptom_node)
workflow.add_node("reasoning_node", reasoning_node)
workflow.add_node("action_node", action_node)
workflow.add_node("response_node", response_node)
workflow.add_node("followup_node", followup_node)

workflow.set_entry_point("intake_node")

workflow.add_conditional_edges("intake_node", route_after_intake)
workflow.add_conditional_edges("symptom_node", route_after_symptom)
workflow.add_conditional_edges("reasoning_node", route_after_reasoning)
workflow.add_conditional_edges("action_node", route_after_action)
workflow.add_conditional_edges("response_node", route_after_response)
workflow.add_edge("followup_node", END)

# Compile the graph
agent_executor = workflow.compile()
