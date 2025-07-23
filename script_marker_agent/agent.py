import google.adk.agents as adk_agents
import google.generativeai as genai
import json

class ContentAnalysisAgent(adk_agents.Agent):
    """
    An agent that analyzes refined student text against a rubric and provides per-criterion scores, justifications, and evidence highlights.
    """
    def __init__(self, name: str = "ContentAnalysisAgent", model: str = "gemini-1.5-flash-latest"):
        super().__init__(name=name)
        self._llm = genai.GenerativeModel(model)

    def run(self, refined_text: str, rubric_markdown: str, question_text: str = None) -> dict:
        """
        Args:
            refined_text: The student's response text (already OCR-refined).
            rubric_markdown: The rubric as a Markdown table.
            question_text: (Optional) The original question text, if available.
        Returns:
            A dict with keys: 'scores', 'justifications', 'evidence', 'total_score'.
        """
        prompt = f"""
You are an expert teacher and assessment assistant. Your job is to analyze a student's response against a rubric and provide detailed, criterion-based feedback.

**Instructions:**
1. Parse the rubric table below. Identify each criterion and its maximum score.
2. For each criterion, analyze the student's response and:
   - Assign a score (0 to max for that criterion).
   - Provide a short justification for the score.
   - Highlight evidence from the student's response (quote or text span).
3. Only analyze the student's response, not the question text.
4. Return your output as a JSON object with this structure:
{{
  "scores": {{ "criterion1": score, ... }},
  "justifications": {{ "criterion1": "...", ... }},
  "evidence": {{ "criterion1": "quoted evidence", ... }},
  "total_score": total
}}
5. Do not include any extra explanation or text outside the JSON object.

**Rubric Table (Markdown):**
---
{rubric_markdown}
---

**Student Response:**
---
{refined_text}
---
"""
        if question_text:
            prompt += f"\n(For reference, the original question was: {question_text})\n"
        try:
            response = self._llm.generate_content(prompt)
            # Extract the JSON object from the response
            text = response.text.strip()
            # Try to find the first and last curly braces to extract JSON
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                json_str = text[start:end+1]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON object found in LLM response.")
        except Exception as e:
            print(f"An error occurred during content analysis: {e}")
            return {"error": str(e), "raw_response": response.text if 'response' in locals() else None}
