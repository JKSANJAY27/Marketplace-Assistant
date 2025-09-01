from google.adk.agents import LlmAgent, SequentialAgent
from pydantic import BaseModel, Field
# from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from typing import Optional
from google.genai import types

# ----------------- Callbacks & Tools -----------------

# Define a callback to request user approval before finalizing content.
# This ensures the artisan has the final say on the generated output.
# def request_user_approval(callback_context: CallbackContext, agent_output: any) -> Optional[LlmResponse]:
#     """Callback to request user approval for the generated content."""
#     # Check if the session is in a "review" state
#     if "review_required" in callback_context.session.state and callback_context.session.state["review_required"]:
#         print(f"[Callback] Requesting user approval for generated content.")
#         # In a real system, this would send a notification to the UI.
#         # For now, we return a direct response asking for approval.
#         return LlmResponse(
#             content=types.Content(
#                 role="model",
#                 parts=[types.Part(text="📝 **Content Ready for Review:** I've prepared a draft of your video ad script and visuals. Please review them. Are you happy with this content, or would you like to make changes?")]
#             )
#         )
#     return None

class ImageGenerationTool:
    """A conceptual tool for generating media assets for marketing.
    In a real system, this would be an API call to a service like Google's Imagen."""
    def __init__(self):
        pass

    def generate_image(self, description: str, style: str) -> str:
        """Generates a high-quality image based on a text description."""
        print(f"[Tool] Generating an image for '{description}' in '{style}' style...")
        # Placeholder for an actual API call.
        return f"Image_URL_for_{description.replace(' ', '_')}"

# Initialize our conceptual tool
image_generator = ImageGenerationTool()

# ----------------- Data Models -----------------

class ProductData(BaseModel):
    """Pydantic model to define the structured output for product details."""
    product_name: str = Field(description="Name of the artisan product.")
    craft_type: str = Field(description="Type of craft (e.g., 'pottery', 'Kalamkari art', 'handloom saree').")
    unique_selling_points: list[str] = Field(description="List of unique features or benefits (e.g., 'handmade', 'sustainable materials', 'cultural heritage').")
    brand_tone: Optional[str] = Field(description="The desired brand tone (e.g., 'elegant', 'rustic', 'modern').")

# ----------------- Sub-Agents -----------------

# Sub-Agent 1: Product Details Extraction Agent
product_extractor_agent = LlmAgent(
    name="ProductExtractorAgent",
    model="gemini-2.0-flash",
    description="Extracts key product details from the artisan's query.",
    instruction="""
        You are a product information extraction expert. Your task is to analyze the user's query and identify all relevant details about their product, including its name, craft type, unique selling points, and brand tone.
        Focus solely on extracting factual information from the conversation.

        User query: '{session.state.user_query}'

        Output only in JSON. The JSON must strictly adhere to the ProductData Pydantic schema.
    """,
    output_schema=ProductData,
    output_key="extracted_product_data"
)

# Sub-Agent 2: Video Ad Script & Visuals Agent
video_ad_agent = LlmAgent(
    name="VideoAdAgent",
    model="gemini-2.0-flash",
    description="Generates a video ad script and suggests visuals based on product details.",
    instruction="""
        You are a creative video marketing specialist. Based on the extracted product details, your task is to:
        1. Write a compelling, short video ad script.
        2. Suggest visual shots or scenes to accompany each part of the script.
        3. Ensure the script and visuals align with the specified brand tone.
        
        Extracted Product Data: {extracted_product_data}
        
        Your response should be formatted to be easily understood and used by a video editor.
    """,
    output_key="video_ad_content",
    # before_response_callback=request_user_approval
)

# Sub-Agent 3: Social Media Post Generator Sub-agent
social_media_agent = LlmAgent(
    name="SocialMediaAgent",
    model="gemini-2.0-flash",
    description="Generates social media captions and hashtags based on the video content.",
    instruction="""
        You are a social media copywriter. Your task is to create a catchy Instagram caption and a set of relevant hashtags based on the generated video ad content.

        Video Ad Script: {video_ad_content}

        The caption should be engaging and the hashtags should help the post reach new audiences.
    """,
    output_key="social_media_post"
)

# ----------------- Main Agent (The Workflow) -----------------

# The main Marketing & Content Creation Agent is a SequentialAgent that orchestrates the workflow.
marketing_agent = SequentialAgent(
    name="MarketingAgent",
    description="Orchestrates the creation of marketing content for artisans.",
    sub_agents=[
        product_extractor_agent,
        video_ad_agent,
        social_media_agent
    ],
)