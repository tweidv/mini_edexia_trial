import google.adk.agents as adk_agents
import google.generativeai as genai
import json
import base64
import traceback
import re

class DiagramMarkingAgent(adk_agents.Agent):
    """
    An agent that analyzes a diagram image and marks it against a rubric using Gemini 1.5 Pro.
    """
    def __init__(self, name: str = "DiagramMarkingAgent", model: str = "gemini-1.5-pro-preview-0409"):
        super().__init__(name=name)
        self._llm = genai.GenerativeModel(model)

    def run(self, image_b64: str, rubric_text: str) -> dict:
        """
        Runs the diagram marking process using Gemini 1.5 Pro.
        Args:
            image_b64: Base64-encoded PNG or JPEG image string.
            rubric_text: The text of the rubric to mark against.
        Returns:
            dict: Marking results, or None if failed.
        """
        try:
            prompt = f"""
            You are an expert in analyzing diagrams. Evaluate the student's diagram image (provided as an image input) based on the rubric below.
            For each criterion, provide:
            - A suggested mark.
            - Detailed reasoning.
            - A text description for the evidence.
            - A bounding box for any incorrect or highlighted element (as [x_min, y_min, x_max, y_max], normalized 0-1).
            - A base64-encoded PNG mask (semi-transparent, highlighting the relevant area).
            Rubric:
            ---
            {rubric_text}
            ---
            Output format (JSON):
            {{
              "criteria": [
                {{
                  "name": "...",
                  "mark": "...",
                  "reasoning": "...",
                  "evidence": "...",
                  "box_2d": [0.0, 0.0, 1.0, 1.0],
                  "mask_png_base64": "..."
                }}
              ]
            }}
            """
            from vertexai.preview.generative_models import Part
            image_bytes = base64.b64decode(image_b64)
            image_part = Part.from_data(data=image_bytes, mime_type="image/png")
            response = self._llm.generate_content([
                image_part,
                prompt
            ])
            # Try to parse JSON from the response
            try:
                # Try to extract JSON from the response text
                match = re.search(r'\{.*\}', str(response), re.DOTALL)
                if match:
                    json_str = match.group(0)
                    result = json.loads(json_str)
                    return result
                else:
                    return None
            except Exception as parse_exc:
                return None
        except Exception as exc:
            traceback.print_exc()
            return None 