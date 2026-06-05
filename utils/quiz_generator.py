import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_quiz(chunks: list, num_questions: int = 5) -> list:
    context = "\n\n".join([chunk for chunk in chunks[:10]])

    prompt = f"""You are an expert quiz creator for academic study.

Your job is to create {num_questions} multiple choice questions that test understanding of the MAIN CONCEPTS and KNOWLEDGE in the document.

STRICT RULES:
- ONLY create questions about the actual subject matter, theories, definitions, formulas, and concepts
- NEVER create questions about: exam rules, number of questions, page numbers, allowed materials, time limits, instructions, document structure, or any administrative information
- Questions must test conceptual understanding, NOT document metadata
- Each question must have exactly 4 options (A, B, C, D)
- Only one correct answer per question
- The correct answer must be directly from the document content

BAD question examples (NEVER do this):
- "How many questions are in this paper?"
- "What is the time allowed for this exam?"
- "Are calculators allowed?"
- "How many marks is section A worth?"

GOOD question examples (DO this):
- "What does the Central Limit Theorem state?"
- "Which probability distribution is used for rare events?"
- "What is the formula for calculating variance?"

Return ONLY valid JSON array, no explanation, no markdown:
[
  {{
    "question": "...",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "answer": "A) ..."
  }}
]

Document Content:
{context}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2048
    )

    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return []