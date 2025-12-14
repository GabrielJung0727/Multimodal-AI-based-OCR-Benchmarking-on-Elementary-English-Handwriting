"""
Shared prompts and schema hints for OCR + judgment tasks.
"""

PROMPT_VERSION = "ocr_and_judge_v1"

# OCR-only prompt (all datasets)
OCR_PROMPT = """
You are an OCR agent. Extract visible text from the image as faithfully as possible.
- Do not guess missing text.
- Preserve line breaks if visible.
- Respond ONLY with JSON matching this schema:
{
  "ocr_text": "string"
}
"""

# Education judgment prompt (synthetic_edu focus)
JUDGE_PROMPT = """
You are an assistant evaluating elementary English handwriting answers.
Steps:
1) Extract the question and the student's answer from the image.
2) If an answer_key is provided separately, compare against it. If not, make your best judgment.
3) Return a JSON object with the required fields.
Rules:
- Do not hallucinate extra questions or answers.
- Confidence must be a float between 0 and 1.
- Keep reason_short to one sentence.
Schema:
{
  "ocr_text": "string",
  "extracted_fields": {
    "question": "string",
    "student_answer": "string"
  },
  "judgement": {
    "predicted_correct": true,
    "confidence": 0.0,
    "reason_short": "string"
  }
}
Respond ONLY with JSON.
"""
