from typing import List, Dict
from openai import OpenAI
import config as C

SYSTEM_PROMPT = """Ты — помощник магистрантов «Цифровые методы в гуманитарных исследованиях».
Используй только предоставленный контекст. Отвечай кратко, структурированно, без выдумок.
В конце укажи источники (section/source). Язык — русский.
"""

def _openai_client() -> OpenAI:
    if not C.OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in environment")
    return OpenAI(api_key=C.OPENAI_API_KEY)

def format_context(ctxs: List[Dict]) -> str:
    blocks = []
    for c in ctxs:
        header = f"[section: {c.get('section')}; source: {c.get('source')}; score: {c.get('score')}]"
        blocks.append(header + "\n" + c.get("text",""))
    return "\n\n---\n\n".join(blocks)

async def generate_answer(user_msg: str, ctxs: List[Dict]) -> str:
    client = _openai_client()
    prompt = (
        f"Вопрос пользователя: {user_msg}\n\n"
        f"Контекст:\n{format_context(ctxs)}\n\n"
        f"Инструкция: ответь по делу, предложи следующие шаги. В конце перечисли источники (section/source)."
    )
    resp = client.chat.completions.create(
        model=C.GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=800
    )
    return resp.choices[0].message.content.strip()
