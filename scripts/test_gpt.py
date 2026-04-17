# test_openai.py
from openai import OpenAI

client = OpenAI()
resp = client.responses.create(
    model="gpt-5.4-mini",
    input="Reply with ONLY OK"
)
print(resp.output_text)