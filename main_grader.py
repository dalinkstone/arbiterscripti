# main_grader.py

import gradio as gr
import docx
import asyncio
from concurrent.futures import ThreadPoolExecutor
from google import generativeai as genai
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os
from pathlib import Path
import time

# --- Configuration and Prompts ---

# This rubric is embedded in the prompt sent to the Gemini API.
GRADING_RUBRIC = """
### NURSING ESSAY GRADING RUBRIC

1.  **Content & Analysis (40 points):**
    * **Thesis/Argument:** Clear, focused, and relevant thesis statement. (10 points)
    * **Evidence & Support:** Strong use of credible, evidence-based sources to support claims. (15 points)
    * **Critical Thinking:** Demonstrates deep analysis, synthesis of ideas, and understanding of complex health science concepts. (15 points)

2.  **Organization & Structure (20 points):**
    * **Introduction:** Engaging introduction with a clear purpose and thesis. (5 points)
    * **Body Paragraphs:** Logical flow, clear topic sentences, and well-developed paragraphs. (10 points)
    * **Conclusion:** Effectively summarizes the argument and provides a sense of closure. (5 points)

3.  **APA Formatting & Citations (25 points):**
    * **Reference List:** Correctly formatted according to APA 7th edition. (10 points)
    * **In-Text Citations:** Accurate and correctly placed in-text citations. (10 points)
    * **General Formatting:** Correct title page, running head, font, and margins. (5 points)

4.  **Clarity & Mechanics (15 points):**
    * **Grammar & Spelling:** Free of significant errors in grammar, punctuation, and spelling. (5 points)
    * **Sentence Structure:** Clear, varied, and concise sentence structure. (5 points)
    * **Professional Tone:** Maintains a scholarly and professional tone appropriate for nursing. (5 points)
"""

# The main prompt for the Gemini API.
# It instructs the AI on its role, the rubric, and the required JSON output format.
GEMINI_PROMPT = f"""
You are an expert-level college university grader for a nursing department. Your task is to evaluate a short health science essay based on a strict rubric and provide your feedback in a structured JSON format.

**DO NOT** provide any introductory text, conversational pleasantries, or explanations outside of the requested JSON structure. Your entire response must be a single, valid JSON object.

**Use the following rubric to grade the essay:**
{GRADING_RUBRIC}

**Instructions:**
1.  Read the entire essay provided below.
2.  Assess the essay against each category in the rubric.
3.  Calculate the total points lost and the final grade out of 100.
4.  Provide brief, specific comments explaining why points were deducted in each category.
5.  Write a 2-3 sentence summary of the overall grade.
6.  Format your entire output as a single JSON object with the following keys and value types:
    - `finalGrade`: (Integer) The final score from 0-100.
    - `pointDeductions`: (Object) An object where keys are the main rubric categories ("Content & Analysis", "Organization & Structure", "APA Formatting & Citations", "Clarity & Mechanics") and values are the integer number of points lost for that category.
    - `feedback`: (Object) An object with the same keys as `pointDeductions`, where values are brief string comments explaining the point deductions for that category. If no points are lost, the comment should be "No points deducted."
    - `summary`: (String) A 2-3 sentence summary of the paper's performance and the rationale for the grade.

**Example of the required JSON output format:**
{{
  "finalGrade": 88,
  "pointDeductions": {{
    "Content & Analysis": 2,
    "Organization & Structure": 0,
    "APA Formatting & Citations": 8,
    "Clarity & Mechanics": 2
  }},
  "feedback": {{
    "Content & Analysis": "The thesis was slightly unfocused, but the evidence used was strong.",
    "Organization & Structure": "No points deducted.",
    "APA Formatting & Citations": "Multiple errors in the reference list formatting and three missing in-text citations.",
    "Clarity & Mechanics": "Minor grammatical errors and occasional awkward phrasing."
  }},
  "summary": "This is a strong paper with excellent critical analysis. The final grade was primarily impacted by significant APA formatting errors, which should be the main focus for improvement."
}}

---
**ESSAY TO GRADE:**

"""


# --- Data Structures ---

@dataclass
class GradingResult:
    """Holds the structured result of a single graded essay."""
    file_name: str
    success: bool
    grade: Optional[int] = None
    deductions: Dict[str, int] = field(default_factory=dict)
    feedback: Dict[str, str] = field(default_factory=dict)
    summary: Optional[str] = None
    error_message: Optional[str] = None


# --- Core Logic Classes ---

class EssayParser:
    """Parses text content from a .docx file."""
    @staticmethod
    def parse_docx(file_path: str) -> str:
        """Extracts all text from a Word document."""
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs if para.text])
        except Exception as e:
            # Handles cases where the file is corrupted or not a valid docx
            raise IOError(f"Could not read file: {os.path.basename(file_path)}. Error: {e}")


class GeminiGrader:
    """Manages interaction with the Google Gemini API for grading."""
    def __init__(self, api_key: str):
        """Initializes the Gemini model."""
        try:
            genai.configure(api_key=api_key)
            # Configuration for safer, more deterministic output
            generation_config = {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
            }
            # Safety settings to prevent the model from refusing to grade
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-pro-latest",
                generation_config=generation_config,
                safety_settings=safety_settings
            )
        except Exception as e:
            raise ValueError(f"Failed to configure Gemini API: {e}")

    def grade_essay(self, essay_text: str, file_name: str) -> GradingResult:
        """
        Sends the essay to Gemini for grading and parses the JSON response.
        This is a synchronous method designed to be run in a thread pool.
        """
        prompt_with_essay = f"{GEMINI_PROMPT}\n{essay_text}"
        try:
            response = self.model.generate_content(prompt_with_essay)
            # Clean the response to ensure it's valid JSON
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
            data = json.loads(cleaned_response)
            
            # Validate the structure of the returned JSON
            required_keys = ["finalGrade", "pointDeductions", "feedback", "summary"]
            if not all(key in data for key in required_keys):
                raise KeyError("The model's response was missing one or more required keys.")

            return GradingResult(
                file_name=file_name,
                success=True,
                grade=data["finalGrade"],
                deductions=data["pointDeductions"],
                feedback=data["feedback"],
                summary=data["summary"]
            )
        except json.JSONDecodeError:
            return GradingResult(
                file_name=file_name,
                success=False,
                error_message="Failed to parse the model's response. The output was not valid JSON."
            )
        except Exception as e:
            return GradingResult(
                file_name=file_name,
                success=False,
                error_message=f"An API or model error occurred: {str(e)}"
            )


# --- Gradio Application ---

async def grade_papers_concurrently(
    files: List[gr.File], api_key: str, progress=gr.Progress(track_tqdm=True)
) -> (str, str):
    """
    The main asynchronous function that orchestrates the grading process.
    It's triggered by the Gradio button click.
    """
    start_time = time.time()
    
    if not api_key:
        raise gr.Error("Google API Key is required.")
    if not files:
        raise gr.Error("Please upload at least one Word document.")

    try:
        grader = GeminiGrader(api_key)
    except ValueError as e:
        raise gr.Error(str(e))

    file_paths = [file.name for file in files]
    total_files = len(file_paths)
    
    # Use a ThreadPoolExecutor to run synchronous tasks concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Create a future for each file processing task
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                executor,
                process_single_file,
                file_path,
                grader
            )
            for file_path in file_paths
        ]

        results = []
        # Process results as they are completed
        for i, future in enumerate(asyncio.as_completed(tasks)):
            progress(i + 1, desc=f"Grading paper {i+1}/{total_files}...")
            result = await future
            results.append(result)

    # --- Format the final output ---
    successful_grades = [res for res in results if res.success]
    failed_grades = [res for res in results if not res.success]
    
    output_markdown = ""
    for result in successful_grades:
        output_markdown += f"### ✅ Grade for: **{result.file_name}**\n"
        output_markdown += f"**Final Grade:** {result.grade}/100\n\n"
        
        # Format point deductions
        deductions_str = ""
        for category, points in result.deductions.items():
            if points > 0:
                deductions_str += f"- **{category}:** Lost {points} points. *Reason: {result.feedback.get(category, 'N/A')}*\n"
        if not deductions_str:
            deductions_str = "Excellent work! No points were deducted.\n"
        
        output_markdown += "**Point Deductions Breakdown:**\n" + deductions_str + "\n"
        output_markdown += f"**Summary:** {result.summary}\n"
        output_markdown += "---\n"

    if failed_grades:
        output_markdown += "### ❌ Failed Papers\n"
        for result in failed_grades:
            output_markdown += f"- **File:** {result.file_name}\n"
            output_markdown += f"  - **Error:** {result.error_message}\n"
        output_markdown += "---\n"
    
    end_time = time.time()
    runtime = f"Total runtime: {end_time - start_time:.2f} seconds."
    
    status = (
        f"Grading complete. {len(successful_grades)} papers graded successfully, "
        f"{len(failed_grades)} failed."
    )
    
    return output_markdown, f"{status}\n{runtime}"


def process_single_file(file_path: str, grader: GeminiGrader) -> GradingResult:
    """
    Synchronous wrapper function to parse and grade one file.
    This function is what runs in each thread of the ThreadPoolExecutor.
    """
    file_name = os.path.basename(file_path)
    try:
        essay_text = EssayParser.parse_docx(file_path)
        if not essay_text.strip():
            return GradingResult(
                file_name=file_name,
                success=False,
                error_message="The document is empty or contains no readable text."
            )
        return grader.grade_essay(essay_text, file_name)
    except Exception as e:
        return GradingResult(file_name=file_name, success=False, error_message=str(e))


# --- Build the Gradio Interface ---

with gr.Blocks(theme=gr.themes.Soft(), title="Nursing Essay Grader") as demo:
    gr.Markdown(
        """
        # 📝 Gemini-Powered Nursing Essay Grader
        Upload one or more student essays in Word format (`.docx`) to have them graded by AI.
        1.  Enter your Google API Key (enabling the Gemini API in your Google Cloud project is required).
        2.  Upload the `.docx` files.
        3.  Click "Grade All Papers". The results will appear below.
        """
    )

    with gr.Row():
        api_key_input = gr.Textbox(
            label="Google API Key",
            placeholder="Enter your Google API Key here",
            type="password",
            scale=1
        )
    
    file_uploads = gr.File(
        label="Upload Word Document Essays",
        file_count="multiple",
        file_types=[".docx"],
        type="filepath" # Use filepath for easier handling
    )
    
    grade_button = gr.Button("🚀 Grade All Papers", variant="primary")
    
    gr.Markdown("---")
    gr.Markdown("## 📊 Grading Results")

    results_output = gr.Markdown(label="Formatted Grades")
    
    status_output = gr.Textbox(
        label="Runtime Status",
        lines=2,
        interactive=False
    )
    
    grade_button.click(
        fn=grade_papers_concurrently,
        inputs=[file_uploads, api_key_input],
        outputs=[results_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
