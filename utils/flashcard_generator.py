import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_flashcards(chunks: list, num_cards: int = 10) -> list:
    context = "\n\n".join([chunk for chunk in chunks[:10]])

    prompt = f"""You are an expert study flashcard creator for academic content.

Your job is to create {num_cards} flashcards that help students learn the MAIN CONCEPTS from the document.

STRICT RULES:
- ONLY create flashcards about actual subject matter: definitions, theories, formulas, concepts, processes, and key facts
- NEVER create flashcards about: exam rules, number of questions, page numbers, allowed materials, time limits, instructions, or any administrative/meta information
- Front: a clear concept question or key term from the subject
- Back: the accurate explanation or definition from the document

BAD flashcard examples (NEVER do this):
- Front: "How many sections does this exam have?"
- Front: "What materials are allowed in the exam?"
- Front: "How much time is given for this paper?"

GOOD flashcard examples (DO this):
- Front: "What is Bayes' Theorem?"
- Front: "Define Standard Deviation"
- Front: "What is the difference between Type I and Type II errors?"

Return ONLY valid JSON array, no explanation, no markdown:
[
  {{
    "front": "...",
    "back": "..."
  }}
]

Document Content:
{context}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2048
    )

    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return []