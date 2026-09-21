import os
from dotenv import load_dotenv

load_dotenv()

from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("\n".join([m.id for m in client.models.list().data]))