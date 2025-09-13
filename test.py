from dotenv import load_dotenv
import os

load_dotenv()
print("Groq Key:", os.getenv("GROQ_API_KEY"))
