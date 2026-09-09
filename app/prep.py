from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv('GROQ_API_KEY')
print('KEY:', repr(key))
print('LENGTH:', len(key) if key else 0)

from groq import Groq
client = Groq()
r = client.chat.completions.create(
    model='openai/gpt-oss-20b',
    messages=[{'role': 'user', 'content': 'hi'}]
)
print(r.choices[0].message.content)