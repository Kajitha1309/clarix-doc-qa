"""
qa_engine.py
------------
Advanced RAG answer generation using prompt engineering.

Techniques Applied:
1. System + User role separation
2. Dynamic response formatting (adapts to question type)
3. Anti-hallucination with strict grounding
4. Exact data extraction for numbers/names/dates
5. Natural conversational tone — no rigid templates
6. Chain-of-thought reasoning (internal, not shown to user)
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_answer(question: str, context_chunks: list) -> dict:
    """
    Generate natural, accurate answers from document context.

    Prompt Engineering Techniques:
    - Role definition: acts like an expert assistant
    - Grounding: strictly uses document content only
    - Format adaptation: style changes based on question type
    - Anti-template: no rigid structure forced on every answer
    - Exact extraction: numbers, names, dates copied precisely
    - Hallucination prevention: explicit rule to not guess

    Args:
        question: User's question string
        context_chunks: Top-k relevant chunks from FAISS

    Returns:
        Dict with natural answer and source chunks
    """

    # Clean context — no section labels shown to LLM
    # This prevents "Section X" appearing in answers
    context_text = "\n\n".join([
        chunk["text"] for chunk in context_chunks
    ])

    system_prompt = """You are an intelligent document assistant — like Claude or ChatGPT.
You help users understand documents by answering their questions naturally and accurately.

CORE RULES (never break these):
1. Answer ONLY from the document content provided — never use outside knowledge
2. If the information is not in the document, say: "I couldn't find that information in this document."
3. Never guess, assume, or make up any information
4. For exact data (names, numbers, dates, phone numbers, GPA, percentages, formulas) — copy them EXACTLY as they appear

RESPONSE STYLE (this is critical):
- Answer naturally like a helpful human expert — NOT like a template-filling machine
- NEVER start with "The answer is found in Section..." or mention sections at all
- NEVER use the same format for every answer — adapt to what was asked:

  → Simple fact question ("What is her GPA?", "What is the candidate name?")
     Just answer directly: "Her GPA is 3.0/4.0" — short and clean

  → List question ("What skills does she have?", "What are the topics?")
     Use clean bullet points naturally

  → Explanation question ("What is machine learning?", "Explain PCA")
     Give a clear explanation using document content, with examples if available

  → Summary request ("Summarize this document")
     Give a well-organized summary with key points

  → Comparison or analysis question
     Structure it logically with clear reasoning

- Be concise — don't pad answers with unnecessary phrases
- Don't repeat the question back in the answer
- Use "According to the document" or "The document states" sparingly — only once if needed
- Write in clear, readable English"""

    user_prompt = f"""Here is the document content:

{context_text}

---

User question: {question}

Answer the question naturally and accurately based on the document content above:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.1,   # Low = accurate, slight variation for natural tone
        max_tokens=1500
    )

    return {
        "answer"       : response.choices[0].message.content,
        "source_chunks": context_chunks
    }