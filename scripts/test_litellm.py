# test_litellm.py
from litellm import completion

resp = completion(
    model="openai/gpt-5.4-mini",
    messages=[{"role": "user", "content": "Reply with ONLY OK"}],
)
print(resp.choices[0].message.content)