import google.adk.agents as adk_agents
import google.generativeai as genai
import json

class MainMarkerAgent(adk_agents.Agent):
    """
    Outputs a clear, structured, human-readable breakdown for each criterion (not JSON, just well-structured text).
    """
    def __init__(self, name: str = "MainMarkerAgent", model: str = "gemini-1.5-flash-latest"):
        super().__init__(name=name)
        self._llm = genai.GenerativeModel(model)

    def run(self, rubric_markdown: str, student_text: str) -> str:
        prompt = f"""
You are an expert teacher and assessment assistant. Analyze the student's work below against the rubric. For each criterion, provide a clear, concise, and structured breakdown in this format:

Criterion: <criterion name>
Suggested Mark: <mark> / <max mark> Marks
Reasoning: <reasoning>
Evidence + Justification: <evidence/quote/justification>

Repeat for each criterion. Do not use JSON, code fences, or Markdown tables. Just output the breakdown for each criterion, one after another, in the format above. Always include the maximum mark for each criterion as shown, using the rubric table to determine the max mark.

**Rubric Table (Markdown):**
---
{rubric_markdown}
---

**Student Work (Text):**
---
{student_text}
---
"""
        response = self._llm.generate_content(prompt)
        return response.text.strip()

class MarkExtractionAgent(adk_agents.Agent):
    """
    Extracts the suggested mark for a criterion from the main agent's output.
    """
    def __init__(self, name: str = "MarkExtractionAgent", model: str = "gemini-1.5-flash-latest"):
        super().__init__(name=name)
        self._llm = genai.GenerativeModel(model)

    def run(self, main_output: str, criterion: str) -> str:
        prompt = f"""
Given the following breakdown for all criteria, extract ONLY the suggested mark for the criterion named '{criterion}'.

Breakdown:
---
{main_output}
---

Return only the mark in the format 'X / Y Marks', where X is the awarded mark and Y is the maximum mark for this criterion. Do not return anything else.
"""
        response = self._llm.generate_content(prompt)
        return response.text.strip()

class ReasoningExtractionAgent(adk_agents.Agent):
    """
    Extracts the reasoning for a criterion from the main agent's output.
    """
    def __init__(self, name: str = "ReasoningExtractionAgent", model: str = "gemini-1.5-flash-latest"):
        super().__init__(name=name)
        self._llm = genai.GenerativeModel(model)

    def run(self, main_output: str, criterion: str) -> str:
        prompt = f"""
Given the following breakdown for all criteria, extract ONLY the reasoning for the criterion named '{criterion}'.

Breakdown:
---
{main_output}
---

Return only the reasoning, nothing else. Do NOT start your response with 'Reasoning:' or any label.
"""
        response = self._llm.generate_content(prompt)
        return response.text.strip()

class EvidenceExtractionAgent(adk_agents.Agent):
    """
    Extracts the evidence/justification for a criterion from the main agent's output.
    """
    def __init__(self, name: str = "EvidenceExtractionAgent", model: str = "gemini-1.5-flash-latest"):
        super().__init__(name=name)
        self._llm = genai.GenerativeModel(model)

    def run(self, main_output: str, criterion: str) -> str:
        prompt = f"""
Given the following breakdown for all criteria, extract ONLY the evidence and justification for the criterion named '{criterion}'.

Breakdown:
---
{main_output}
---

Return only the evidence and justification, nothing else. Do NOT start your response with 'Evidence + Justification:' or any label.
"""
        response = self._llm.generate_content(prompt)
        return response.text.strip()

class RubricCriterionExtractorAgent(adk_agents.Agent):
    """
    An agent that extracts a clean list of criterion names from a Markdown rubric table, ignoring separator lines and formatting artifacts.
    """
    def __init__(self, name: str = "RubricCriterionExtractorAgent", model: str = "gemini-1.5-flash-latest"):
        super().__init__(name=name)
        self._llm = genai.GenerativeModel(model)

    def run(self, rubric_markdown: str) -> list:
        prompt = f"""
You are an expert at parsing Markdown tables. Given the following Markdown rubric table, extract a clean Python list of only the actual criterion names (ignore separator lines, formatting artifacts, asterisks, dashes, colons, etc.).

Return only a valid Python list of strings, nothing else.

**Rubric Table:**
---
{rubric_markdown}
---
"""
        try:
            response = self._llm.generate_content(prompt)
            text = response.text.strip()
            # Try to extract a Python list from the response
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1:
                list_str = text[start:end+1]
                return json.loads(list_str.replace("'", '"'))
            else:
                raise ValueError("No Python list found in LLM response.")
        except Exception as e:
            print(f"An error occurred during criterion extraction: {e}")
            return []

class HighlightExtractionAgent(adk_agents.Agent):
    """
    Extracts a list of single words from the student work that should be highlighted as evidence for a criterion, based on the evidence+justification text.
    """
    def __init__(self, name: str = "HighlightExtractionAgent", model: str = "gemini-1.5-flash-latest"):
        super().__init__(name=name)
        self._llm = genai.GenerativeModel(model)

    def run(self, student_text: str, evidence_justification: str) -> list:
        prompt = f"""
Given the following student work and the evidence+justification for a criterion, return a Python list of the exact single words from the student work that best serve as evidence for the mark. Only include words that appear in the student work. Return only a Python list of strings, nothing else.

Student Work:
---
{student_text}
---

Evidence + Justification:
---
{evidence_justification}
---
"""
        response = self._llm.generate_content(prompt)
        text = response.text.strip()
        # Try to extract a Python list from the response
        try:
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1:
                list_str = text[start:end+1]
                return json.loads(list_str.replace("'", '"'))
            else:
                raise ValueError("No Python list found in LLM response.")
        except Exception as e:
            print(f"[HighlightExtractionAgent] Error parsing LLM response: {e}\nRaw response: {text}")
            return []
