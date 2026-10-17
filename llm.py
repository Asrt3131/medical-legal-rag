from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """تو یک دستیار متخصص اسناد پزشکی/حقوقی هستی.
فقط بر اساس متن‌هایی که در ادامه می‌آید پاسخ بده.
اگر پاسخ در متن نبود، صریح بگو «در اسناد موجود نیست».
همیشه به شماره صفحه منبع ارجاع بده."""

def generate_answer(query: str, sources: list) -> str:
    context = "\n\n---\n\n".join(
        f"[صفحه {s['page']}]\n{s['text']}" for s in sources
    )

    user_prompt = f"""متن‌های مرجع:
{context}

سوال کاربر: {query}

پاسخ:"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content