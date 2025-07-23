import os
import google.adk.agents as adk_agents
import google.generativeai as genai
from PIL import Image
import io
import base64

class OcrRefinementAgent(adk_agents.Agent):
    """
    An agent that refines raw OCR text by correcting errors,
    improving formatting, and using the original image for context.
    """
    def __init__(self, name: str = "OcrRefinementAgent", model: str = "gemini-1.5-flash-latest"):
        super().__init__(name=name)
        self._llm = genai.GenerativeModel(model)

    def run(
        self, raw_ocr_text: str, original_image_path: str
    ) -> str:
        """
        Runs the OCR refinement process.

        Args:
            raw_ocr_text: The raw text from the OCR process.
            original_image_path: Path to the original image for context.

        Returns:
            The refined text as a string.
        """
        print("Starting OCR refinement...")

        try:
            image = Image.open(original_image_path)
        except FileNotFoundError:
            return "Error: Original image not found at path."

        prompt = self._create_prompt(raw_ocr_text)

        print("Sending request to the LLM...")
        response = self._llm.generate_content([prompt, image])
        
        refined_text = response.text
        print("Received refined text from LLM.")
        
        return refined_text

    def _create_prompt(self, raw_ocr_text: str) -> str:
        """Creates the prompt for the LLM."""
        return f"""
You are an expert assistant for a teacher's script-marking tool. Your role is to be a **transcriber**, not a corrector.
Your task is to clean up the raw text from an Optical Character Recognition (OCR) process so it perfectly matches what the student **actually wrote** in the provided image, including any original mistakes.

**Crucial Rule: Do NOT correct the student's spelling or grammar.** If the student spelled a word incorrectly, your output must contain that exact same spelling mistake. Your job is to fix the machine's errors, not the student's errors.

**Instructions:**
1.  **Fix OCR Reading Errors:** The OCR may have misread letters or words (e.g., reading "wale" instead of "wake" when "wake" is clearly written). Use the image to fix these reading errors.
2.  **Restore Missing Text:** The OCR might miss small words (like "a", "the", "I") or punctuation that are clearly visible in the handwriting. Add them back in.
3.  **Transcribe Student Errors Faithfully:** If you see a spelling or grammar mistake in the student's handwriting, transcribe it exactly as it is written. For example, if the student wrote "I think of sucess", your output must be "I think of sucess".
4.  **Preserve Formatting:** Maintain the original structure. Keep paragraph breaks, line breaks, and indentation as they appear in the handwritten document.
5.  **Output Only the Transcribed Text:** Your final output should only be the corrected text. Do not add any extra explanations or introductory phrases.

**Raw OCR Text to Refine:**
---
{raw_ocr_text}
---
"""

def setup_phoenix_tracing():
    """Sets up OpenInference tracing with Phoenix."""
    try:
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor
        import phoenix.trace
        
        # Start the Phoenix server in the background
        # Note: If you already have a Phoenix server running, this will connect to it.
        phoenix.launch_app()

        # Instrument the ADK library
        GoogleADKInstrumentor().instrument()
        
        print("Phoenix tracing for Google ADK has been enabled.")
        print("You can view traces at http://127.0.0.1:6006")

    except ImportError:
        print("Phoenix or OpenInference libraries not found. Skipping tracing setup.")
    except Exception as e:
        print(f"An error occurred during Phoenix setup: {e}")

if __name__ == '__main__':
    # 1. Enable Phoenix Tracing
    setup_phoenix_tracing()

    # 2. Get the raw OCR text (we'll reuse the logic from your Flask app)
    # This part is for demonstration. In a real workflow, this would come from another agent.
    
    # Temporarily add project root to path to import app logic
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app import detect_document_text
    
    # Load environment variables from .env
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))


    test_image_path = "C:/Users/tweid/Projects/mini_edexia_trial/handwriting.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found at '{test_image_path}'. Please check the path.")
    elif not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("GOOGLE_APPLICATION_CREDENTIALS not found. Make sure it's in your .env file.")
    elif not os.getenv("GOOGLE_API_KEY"):
         print("GOOGLE_API_KEY not found. Make sure it's in your .env file.")
    else:
        print("\n--- Running OCR to get raw text ---")
        full_annotation = detect_document_text(test_image_path)
        raw_text = full_annotation.text
        print("Raw Text from OCR:\n", raw_text)
        
        print("\n--- Initializing and running OcrRefinementAgent ---")
        # 3. Initialize and run the agent
        refinement_agent = OcrRefinementAgent()
        refined_text_output = refinement_agent.run(
            raw_ocr_text=raw_text,
            original_image_path=test_image_path
        )
        
        print("\n--- Final Refined Text ---")
        print(refined_text_output) 