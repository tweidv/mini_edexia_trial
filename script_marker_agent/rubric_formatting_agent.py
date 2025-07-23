import google.adk.agents as adk_agents
import google.generativeai as genai

class RubricFormattingAgent(adk_agents.Agent):
    """
    An agent that takes raw rubric text and formats it into a clean Markdown table.
    """
    def __init__(self, name: str = "RubricFormattingAgent", model: str = "gemini-1.5-flash-latest"):
        super().__init__(name=name)
        self._llm = genai.GenerativeModel(model)

    def run(self, raw_rubric_text: str) -> str:
        """
        Runs the rubric formatting process.

        Args:
            raw_rubric_text: The raw text of the rubric from a text box or OCR.

        Returns:
            A string containing the formatted rubric as a Markdown table.
        """
        print("Starting rubric formatting...")

        prompt = f"""
        You are an expert in educational assessment. Your task is to take a raw, unstructured rubric and reformat it into a clean, well-organized Markdown table.

        **Instructions:**
        1. Analyze the provided text to identify the core criteria and all mark ranges or performance levels (e.g., 'Not Met', 'Developing', 'Proficient', etc.).
        2. If the rubric is structured as a table with multiple columns for different mark ranges, preserve all columns and their descriptions in the Markdown table. Do not collapse them into a single description or max score.
        3. You may need to completely reformat the table for clarity and usability. Do not simply copy the original table if it is unclear or poorly structured.
        4. Your Markdown table should have a column for the criterion, a column for each mark range (with the range name as the column header), and a column for the score (if present).
        5. Make it very clear what the maximum score is for each criterion (e.g., in a 'Max Score' column or similar), and also include the maximum total score for the whole task (e.g., as a final row labeled 'Total Possible Score').
        6. If the rubric text is vague, use your expertise to infer the structure. The goal is to create a clear, usable table for a teacher.
        7. Your output must be **only** the Markdown table. Do not include any introductory text, explanations, or code fences (```).

        **Raw Rubric Text:**
        ---
        {raw_rubric_text}
        ---
        """
        try:
            response = self._llm.generate_content(prompt)
            formatted_text = response.text.strip()
            print("Received formatted rubric from LLM.")
            return formatted_text
        except Exception as e:
            print(f"An error occurred during rubric formatting: {e}")
            # As a fallback, return the original text
            return raw_rubric_text 