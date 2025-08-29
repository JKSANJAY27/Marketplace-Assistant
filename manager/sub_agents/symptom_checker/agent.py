from google.adk.agents import LlmAgent, SequentialAgent
from pydantic import BaseModel, Field
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from typing import Optional
from google.genai import types

# ----------------- Callbacks & Tools -----------------

# Define a PII list for a basic PII removal callback
PII_KEYWORDS = ["my name is", "my phone number is", "my address is", "I live at", "my birth date is", "my social security number is"]

def remove_pii(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    """Callback to remove personally identifiable information."""
    prompt = ""
    if llm_request.contents:
        for content in llm_request.contents:
            if content.parts:
                for part in content.parts:
                    if part.text:
                        prompt += part.text + " "
    
    # Simple check and replacement for demonstration.
    # In a real system, you would use a more robust PII detection model.
    sanitized_prompt = prompt
    for keyword in PII_KEYWORDS:
        sanitized_prompt = sanitized_prompt.replace(keyword, "")
    
    if prompt != sanitized_prompt:
        # Overwrite the original request with the sanitized one
        llm_request.contents[0].parts[0].text = sanitized_prompt
        print(f"[Callback] PII removed from user prompt.")

    return None

def emergency_triage_callback(callback_context: CallbackContext, agent_output: any) -> Optional[LlmResponse]:
    """
    Callback that checks the agent's output for critical health conditions.
    This runs after the diagnosis sub-agent's response is generated.
    """
    if "emergency_protocol" in callback_context.session.state:
        # This is a flag set by the Diagnositics & Triage Sub-Agent
        print(f"[Callback] Critical condition detected. Triggering emergency redirect.")
        # Trigger the redirection workflow (e.g., call a GeolocationTool)
        
        # In a real scenario, this would be a tool call, but for now we'll return a direct response.
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text="⚠️ **Emergency Alert:** Your symptoms indicate a critical condition. Please proceed to the nearest emergency room immediately. We have located the nearest hospital for you.")]
            )
        )
    return None

# ----------------- Sub-Agents -----------------

class SymptomDataOutput(BaseModel):
    symptoms: list[str] = Field(description="List of symptoms identified from the user query.")
    severity: str = Field(description="Overall perceived severity (e.g., 'mild', 'moderate', 'severe').")
    duration: Optional[str] = Field(description="Duration of symptoms, if provided.")

# Sub-Agent 1: Symptom & Data Extraction Agent
symptom_extractor_agent = LlmAgent(
    name="SymptomExtractorAgent",
    model="gemini-2.0-flash",
    description="Extracts symptoms, severity, and duration from the user's initial query.",
    instruction="""
        You are a symptom extraction expert. Your task is to analyze the user's query and identify all symptoms, their severity (if mentioned), and the duration of the condition. 
        Focus solely on extracting factual medical information.

        User query: '{session.state.user_query}'

        Output only in JSON. The JSON must strictly adhere to the following format:
        {{
            "symptoms": ["list", "of", "symptoms"],
            "severity": "e.g., 'mild', 'moderate', 'severe', 'unknown'",
            "duration": "e.g., '2 days', 'a week', 'unknown'"
        }}
    """,
    output_schema=SymptomDataOutput,
    output_key="extracted_symptom_data",
    include_contents="default",
    before_model_callback=remove_pii
)

# Sub-Agent 2: Diagnostics & Triage Sub-Agent
diagnostics_triage_agent = LlmAgent(
    name="DiagnosticsAndTriageAgent",
    model="gemini-2.0-flash", # Use a more capable model for diagnostics
    description="Generates a preliminary diagnosis and flags critical conditions.",
    instruction="""
        You are a medical pre-diagnosis AI. Based on the extracted symptoms, your task is to:
        1.  Suggest a list of possible conditions.
        2.  Identify if any symptoms are severe or life-threatening.
        3.  If a critical condition is detected, set a flag for emergency triage.

        Extracted Symptoms: {extracted_symptom_data}

        *DO NOT provide a formal diagnosis. Use cautious language.*

        Your response should be in a conversational and empathetic tone.
        If symptoms are critical (e.g., chest pain, difficulty breathing, sudden loss of consciousness), include the phrase "trigger_emergency_protocol" at the end of your response.
    """,
    # This agent's output will be the conversational response to the user.
    output_key="preliminary_diagnosis_response",
    # before_response_callback=emergency_triage_callback
)

# Sub-Agent 3: Drug Interaction & Safety Sub-Agent
drug_interaction_agent = LlmAgent(
    name="DrugInteractionAgent",
    model="gemini-2.0-flash",
    description="Checks for potential drug interactions based on user-provided information.",
    instruction="""
        You are a drug safety checker. Your task is to analyze user-provided medication information and check for potential conflicts.
        This agent is invoked only if the user mentions taking medications.
        
        Use the following tools to perform checks:
        - `get_allopathic_drug_interactions`
        - `get_ayurvedic_allopathic_interactions`
        
        Based on your findings, provide a clear, easy-to-understand alert to the user.
    """,
    output_key="drug_interaction_alert",
)

# Sub-Agent 4: Response Formatter Sub-Agent
response_formatter_agent = LlmAgent(
    name="ResponseFormatterAgent",
    model="gemini-2.0-flash",
    description="Formats all outputs into a final, user-friendly response.",
    instruction="""
        You are a kind and professional healthcare assistant. Your task is to combine the preliminary diagnosis and any drug interaction alerts into a single, cohesive, and empathetic message for the user.

        Preliminary Diagnosis: {preliminary_diagnosis_response}
        Drug Interaction Alert (if any): {drug_interaction_alert}

        Your response should:
        1. Start with an empathetic greeting.
        2. Present the preliminary diagnosis clearly, using cautious language.
        3. Include any drug interaction alerts as a separate, clearly marked section.
        4. End with a strong recommendation to consult a doctor.

        Ensure the entire message is in a friendly, conversational tone.
    """,
    output_key="final_symptom_checker_response"
)


# Main Sequential Agent for the Symptom Checker Workflow
symptom_checker_pipeline = SequentialAgent(
    name="SymptomCheckerPipeline",
    description="Orchestrates the symptom collection, diagnosis, and safety checks for a user.",
    sub_agents=[
        symptom_extractor_agent,
        diagnostics_triage_agent,
        drug_interaction_agent,
        response_formatter_agent
    ],
)